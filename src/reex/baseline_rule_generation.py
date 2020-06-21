from skrules import SkopeRules
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
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
import time
import gzip
import obonet
import networkx as nx



def read_the_dataset(dataset_name, attribute_mapping = None):

    """
    dataset mora met target atribut, poleg tega ostali so geni.
    """
    
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
    logging.info("Found {} GO maps.".format(len(col_indices)))
    new_dx = rd.iloc[:,col_indices]
    new_dx.columns = col_names
    logging.info("Considering DF of shape {}".format(new_dx.shape))
    return new_dx, target_vector

def read_generic_gaf(gaf_file):

    symmap = {}
    with gzip.open(gaf_file,"rt") as gf:
        for line in gf:
            line = line.strip().split("\t")
            if "UniProt" in line[0] and len(line) > 5:
                symmap[line[2]] = line[4]
    logging.info("Found {} mappings.".format(len(symmap)))
    return symmap
    
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
    logging.info(nx.info(reverseGraph))
    tnum = len(wholeset)
    logging.info("Found {} unique edge types, {}".format(tnum," | ".join(wholeset)))
    return reverseGraph




if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()    
    parser.add_argument("--expression_dataset",default='../data/final_versions/Breast_A.csv', type = str)
    parser.add_argument("--mapping_file",default='../mapping/goa_human.gaf.gz', type = str)
    parser.add_argument("--iterations",default=20, type = int)
    parser.add_argument("--background_knowledge",default='http://purl.obolibrary.org/obo/go/go-basic.obo', type = str)
	
    args = parser.parse_args()
    parsed_dataset, target_vector = read_the_dataset(args.expression_dataset, attribute_mapping = args.mapping_file)
    number_of_iterations = (args.iterations)
    ontology_graph = get_ontology(obo_link = args.background_knowledge)
    
    feature_names = list(parsed_dataset.columns)
    parsed_dataset = np.array(parsed_dataset)
    target_vector = np.array(target_vector)
    #print(parsed_dataset)
    
    for k in range(number_of_iterations):
        #check whether there are any features left
        if len(feature_names) < 1:
            print("Exiting due to lack of features")
            quit()
            
        print("Iteration number:", k)
        #generate rules
        clf = SkopeRules(max_depth_duplication=2,
                         n_estimators=30,
                         precision_min=0.3,
                         recall_min=0.1,
                         feature_names=feature_names)
        
        unique_values = list(set(target_vector))
        if 0 not in unique_values:
            target_vector = target_vector - 1
        

        for idx, species in enumerate(unique_values):
            X, y = parsed_dataset, target_vector
            #print(X)
            #print(y)
            clf.fit(X, y == idx)
            rules = clf.rules_[0:3]
            print("Rules for class:", species)
            for rule in rules:
                print(rule)
            print()
            print(20*'=')
            print()

        #now generalize for one step
        ancestor_list = []
        number_of_descendants = []
        projection_to_ancestors = []
        for feature in feature_names:
            ancestors = [x[0] for x in ontology_graph.in_edges(feature)]
            list_to_append = []
            for ancestor in ancestors:
                if ancestor not in ancestor_list:
                    ancestor_list.append(ancestor)
                    number_of_descendants.append(1)
                else:
                    number_of_descendants[ancestor_list.index(ancestor)] += 1
                list_to_append.append(ancestor_list.index(ancestor))
            projection_to_ancestors.append(list_to_append)
        
        
        #and construct new dataset
        new_dataset = np.empty([parsed_dataset.shape[0],len(ancestor_list)], dtype=float)
        for row in range(parsed_dataset.shape[0]):
            new_row = [0] * len(ancestor_list)
            for column in range(parsed_dataset.shape[1]):
                for item in projection_to_ancestors[column]:
                    new_row[item] += parsed_dataset[row][column]
            #normalize and append
            for c in range(new_dataset.shape[1]):
                new_dataset[row][c] = new_row[c] / number_of_descendants[c]
                
                
        parsed_dataset = new_dataset
        feature_names = ancestor_list
        
        
        print("-----------------------------------")
        print()
        print()
        