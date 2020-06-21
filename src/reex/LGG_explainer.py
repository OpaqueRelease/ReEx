import shap
import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import model_selection as ms
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import utils
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn import svm
#import logging
#logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
#logging.getLogger().setLevel(logging.INFO)
import time
import gzip
import networkx as nx
import obonet
import timeit
import sys
import os


def list_to_ancestor_set(graph, list):
    result = set()
    for iterator in list:
        result = result.union(nx.descendants(graph,iterator))
    return result
    
    
def read_generic_gaf(gaf_file):

    symmap = {}
    with gzip.open(gaf_file,"rt") as gf:
        for line in gf:
            line = line.strip().split("\t")
            if "UniProt" in line[0] and len(line) > 5:
                symmap[line[2]] = line[4]
    #logging.info("Found {} mappings.".format(len(symmap)))
    return symmap
                
def read_the_dataset(dataset_name, attribute_mapping = None):

    gaf_map = read_generic_gaf(attribute_mapping)        
    rd = pd.read_csv(dataset_name)

    ## re-map.
    colx = rd.columns.tolist()
    col_indices = []
    col_names = []
    target_vector = rd['target'].values
    
    for enx, x in enumerate(colx):
        if x in gaf_map:
            col_indices.append(enx)
            col_names.append(gaf_map[x])
    #logging.info("Found {} GO maps.".format(len(col_indices)))
    new_dx = rd.iloc[:,col_indices]
    new_dx.columns = col_names
    #logging.info("Considering DF of shape {}".format(new_dx.shape))
    return new_dx, target_vector

def train_and_explain(X, Y, subset = 100, classifier_index = 0):
    
    """
    A set of calls for obtaining aggregates of explanations.
    """
    ## label encoding
    lab_enc = preprocessing.LabelEncoder()
    training_scores_encoded = lab_enc.fit_transform(Y)

    #logging.info("Feature pre-selection ({}).".format(subset))
    #X = X.iloc[:,1:100]
    minf = mutual_info_classif(X.values, training_scores_encoded)
    top_k = np.argsort(minf)[::-1][0:subset]
    attribute_vector = X.columns[top_k]
    X = X.astype(float).values[:,top_k]    
    skf = StratifiedKFold(n_splits=10)
    performances = []
    enx = 0
    t_start = time.time()
    #logging.info("Starting importance estimation ..  shape: {}".format(X.shape))
    
    per_class_explanations = defaultdict(list)
    
    classifiers = [GradientBoostingClassifier(), RandomForestClassifier(n_estimators=10), svm.SVC(probability=True)]
    
    for train_index, test_index in skf.split(X, Y):        
        enx+=1
        clf = classifiers[classifier_index]
        x_train = X[train_index,:]
        x_test = X[test_index]
        y_train = Y[train_index]
        y_test = Y[test_index]
        model = clf.fit(x_train, y_train)    
        preds = model.predict(x_test)
        if len(np.unique(y_train)) > 1:
            average = "micro"
        perf = f1_score(preds,y_test, average = average)
        performances.append(perf)
        #logging.info("Performance in fold {}, {} (F1)".format(enx, perf))
        explainer = shap.KernelExplainer(model.predict_proba, x_train)
        for unique_class in set(preds):
            cors_neg = np.array([enx for enx, pred_tuple in enumerate(zip(preds, y_test)) if pred_tuple[0] == pred_tuple[1] and pred_tuple[0] == unique_class])
            if cors_neg.size != 0:
                shap_values = explainer.shap_values(x_test[cors_neg], nsamples = 10, verbose = False)
                stack = np.mean(np.vstack(shap_values),axis = 0)
                per_class_explanations[unique_class].append(stack)

    final_explanations = {}
    for class_name, explanation_set in per_class_explanations.items():
        final_explanations[class_name] = np.mean(np.matrix(explanation_set),axis = 0)
                     
    t_end = time.time() - t_start
    #logging.info("Time spent on explanation estimation {}s.".format(t_end))
    
    average_perf = (np.mean(performances), np.std(performances))
    #logging.info("Final performance: {}".format(average_perf))

    return (final_explanations, attribute_vector)


def get_ontology(obo_link = '../ontologies/go-basic.obo'):

    try:
        graph = obonet.read_obo(obo_link)
    except:
        ## naj drugace proba lokalno to prebrat iz obo_link fajla.
        obo_link = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
        graph = obonet.read_obo(obo_link)

    numberOfNodes = graph.number_of_nodes()
    reverseGraph = nx.MultiDiGraph()

    ## generate whole graph first, we'll specialize later.
    wholeset = set()
    for edge in list(graph.edges()):
        edge_info = set(graph.get_edge_data(edge[0], edge[1]).keys())
        wholeset = wholeset.union(edge_info)
        for itype in edge_info:
            reverseGraph.add_edge(edge[1], edge[0], type = itype)                
    #logging.info(nx.info(reverseGraph))
    tnum = len(wholeset)
    #logging.info("Found {} unique edge types, {}".format(tnum," | ".join(wholeset)))
    return reverseGraph


def lgg_multiple_sets(list_of_termsets, ontology, intersectionRatio = 0):

    """
    :param list_of_termsets: termsets as found by explanations
    :param ontology: a nx graph
    :param max_gen: number of iterations in worst case
    """
    
	# vzames prednike in odstranis tiste, ki imajo za naslednike element iz ene izmed ostalih mnozic
    converged = np.ones(len(list_of_termsets))
    tmp_ancestor_storage = None
    ancestor_storage = list_of_termsets.copy()
    
    #speedup
    #trace_sets = list_of_termsets.copy()

    count_iterations = [-1]*len(converged)
    while not all(v == 0 for v in converged):
        for a in range(len(count_iterations)):
            if(converged[a] == 1):
                count_iterations[a] += 1

            
        tmp_ancestor_storage = ancestor_storage.copy()
        for enx, termset in enumerate(ancestor_storage):
                    if converged[enx] == 1:
                        ancestors_ = set([x[0] for x in ontology.in_edges(termset)])
                        ancestor_storage[enx] = ancestors_
                                  
        for ancestor_Set in range(len(ancestor_storage)):
            removalSet = set()
            for val in ancestor_storage[ancestor_Set]:
                #speedup
                #desc_of_val = nx.descendants(ontology,val)
                numberOfTerms = 0
                intersectionCount = 0
                for setTwo in range(len(tmp_ancestor_storage)):
                    #split case for faster performance
                    if intersectionRatio == 0:
                        #speedup
                        #if ancestor_Set != setTwo and len(set.intersection(desc_of_val, list_of_termsets[setTwo])) > 0:
                        if ancestor_Set != setTwo and len(set.intersection(set(val), ancestor_storage[setTwo])) > 0:
                            removalSet.add(val)
                            break
                    else:
                         if ancestor_Set != setTwo:
                            #intersectionCount += len(set.intersection(desc_of_val, list_of_termsets[setTwo]))
                            intersectionCount += len(set.intersection(val, ancestor_storage[setTwo]))
                            numberOfTerms += len(ancestor_storage[setTwo])
                            
                if numberOfTerms != 0 and intersectionRatio > 0 and intersectionRatio < intersectionCount / numberOfTerms:
                    removalSet.add(val)
                    
            newSet = ancestor_storage[ancestor_Set].difference(removalSet)            
            if newSet : 
                ancestor_storage[ancestor_Set] = newSet
            else:               
                ancestor_storage[ancestor_Set] = tmp_ancestor_storage[ancestor_Set]            
                converged[ancestor_Set] = 0
             
    return((ancestor_storage, count_iterations))
        

def print_results(class_names, subsets, evaluation):
    """
    A method which prints class names with the generalized set of terms
    :param class_names: list of class names
    :param subsets: a tuple of a list of generalized term sets and number of generalization iterations
    """
    for i in range(len(class_names)):
         print("RESULT_TAG" + "\t" + str(class_names[i]) + "\t" + str(subsets[0][i]) + "\t" + str(evaluation[i]) + "\t" + str(subsets[1][i]))

    
def evaluate(original, generalized):
    """
    A method which counts how many original terms were generalized in each set
    :param original: list of original term sets
    :param generalized: list of generalized term sets
    """
    result = []
    for i in range(len(original)):
        intersection = [value for value in generalized[i] if value in original[i]] 
        result.append((len(original[i]) - len(intersection)) / len(original[i]))
    return result

def generalizeLGG(ontology_graph, explanations = None, attributes = None, target_relations = {"is_a","partOf"}, test_run = False, intersectionRatio = 0, abs = 0):

    """
    A method which generalizes explanations based on the knowledge graph structure.
    :param ontology_graph: a NetworkX graph.fsampl
    :param term_mapping: a mapping from gene names to GO terms.
    :param explanations: a listdict with explanation data
    :param attributes: attribute names.
    """

    if not explanations:
        test_run = True

    if test_run:
        a = list(ontology_graph.nodes())

        ## two random lists
        t1 = set(a[0:42])
        t2 = set(a[1000:1070])
        ontology_subgraph = nx.DiGraph([x for x in ontology_graph.edges(data = True) if x[2]['type'] in target_relations])
        #logging.info("Created ontology subgraph of properties: \\{}".format(nx.info(ontology_subgraph)))
        subsets = lgg_multiple_sets([t1,t2], ontology_subgraph)
        evaluation = evaluate([t1,t2], subsets[0])
        print_results(["t1", "t2"], subsets, evaluation, intersectionRatio = intersectionRatio)
        
    else:
        term_sets_per_class = []
        class_names = []
        for class_name, explanation_vector in explanations.items():
            #compute the threshold
            if abs == 0:
                greater_than_zero_vector = explanation_vector[explanation_vector > 0]
                if len(greater_than_zero_vector) < 1:
                    print("Zero size feature vector. Aborting...")
                    sys.exit()
                maxVector = np.amax(greater_than_zero_vector)
                
            else:
                greater_than_zero_vector = explanation_vector[np.absolute(explanation_vector) > 0]
                if len(greater_than_zero_vector) < 1:
                    print("Zero size feature vector. Aborting...")
                    sys.exit()
                maxVector = np.amax(np.absolute(greater_than_zero_vector))
            threshold = 0.9 * maxVector
            while True:
                if abs == 0:
                    above_threshold = set(np.argwhere(greater_than_zero_vector >= threshold).flatten())
                else:
                    above_threshold = set(np.argwhere(np.absolute(greater_than_zero_vector) >= threshold).flatten())
                    
                if len(above_threshold) > 5 or threshold < 0.3 * maxVector:
                    break
                threshold *= 0.9
                
            terms = set([x for enx,x in enumerate(attributes) if enx in above_threshold])
            term_sets_per_class.append(terms)
            class_names.append(class_name)
        subsets = lgg_multiple_sets(term_sets_per_class, ontology_graph, intersectionRatio = intersectionRatio)
        
        evaluation = evaluate(term_sets_per_class, subsets[0])
        print("before generalization:")
        for setIndex in range(len(term_sets_per_class)):
            print(class_names[setIndex] + ": " + str(list(term_sets_per_class[setIndex])))
        print_results(class_names, subsets, evaluation)


if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser()    
        parser.add_argument("--expression_dataset",default='../data/TCGA-PANCAN-HiSeq-801x20531/readyDataset.csv', type = str)    
        parser.add_argument("--background_knowledge",default='http://purl.obolibrary.org/obo/go/go-basic.obo', type = str)
        parser.add_argument("--mapping_file",default='../mapping/goa_human.gaf.gz', type = str)
        parser.add_argument("--intersection_ratio",default=0, type = float)
        parser.add_argument("--subset_size", default=100, type = int)
        parser.add_argument("--classifier_index", default=0, type = int)
        parser.add_argument("--absolute",default=0, type = int)
        # ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/
        
       
        
        args = parser.parse_args()
      
        
        ontology_graph = get_ontology(obo_link = args.background_knowledge)    
        parsed_dataset, target_vector = read_the_dataset(args.expression_dataset, attribute_mapping = args.mapping_file)

        explanations, attributes = train_and_explain(parsed_dataset, target_vector, subset = args.subset_size, classifier_index = args.classifier_index)

            
        generalizeLGG(ontology_graph, explanations = explanations, attributes = attributes, test_run = False,  abs = args.absolute, intersectionRatio = args.intersection_ratio)
        #generalizeLGG(ontology_graph, test_run = True)
