## This is the main blob of code comprised of reasoning methods.
import numpy as np
import networkx as nx
import sys
import logging
import random
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
from nltk.corpus import wordnet as wn

try:
    from py3plex.algorithms import hedwig
except:
    logging.info("In order to use Hedwig, pip install py3plex")
    
    
def update_dict_of_generalization_depth(old_dictionary, new_set, iterations_count):
    """
    updates the dictionary that represents generalization depth for each term in this set. If the term is new, it's depth equals the number of iterations, else it stays the same
    """
    new_dict = {}
    for item in new_set:
        if item in old_dictionary.keys():
            new_dict[item] = old_dictionary[item]
        else:
            new_dict[item] = iterations_count
            
    return new_dict
                
    

def selective_staircase_multiple_sets(list_of_termsets, ontology, intersectionRatio = 0):

    """
    :param list_of_termsets: termsets as found by explanations
    :param ontology: a nx graph
    :param intersectionRatio: ratio of connected terms of other classes that is allowed in generalization
    """
    
    converged = np.ones(len(list_of_termsets))
    tmp_ancestor_storage = None
    ancestor_storage = list_of_termsets.copy()

    #count_iterations = [-1]*len(converged)
    
    ## count_iterations_per_term is a list of dictionaries, each dictionary is specific to a class and represents the generalization depth of each element in that class
    count_iterations_per_term = []
    for a in range(len(list_of_termsets)):
        count_iterations_per_term.append({})
        for b in list_of_termsets[a]:
            count_iterations_per_term[a][b] = 0
            
        
    count_iterations = 1    
    while not all(v == 0 for v in converged):
        #for a in range(len(count_iterations)):
          #  if(converged[a] == 1):
           #     count_iterations[a] += 1
       
            
        tmp_ancestor_storage = ancestor_storage.copy()
        
        ## list of lists of dictionaries,  each dictionary contains elements of this class and it's ancestors
        list_of_dicts = []
        for a in range(len(ancestor_storage)):
            list_of_dicts.append({})
            
        for enx in range(len(ancestor_storage)):
            termset = ancestor_storage[enx]
            if converged[enx] == 1:
                for term in termset:
                    list_of_dicts[enx][term] = set([x[0] for x in ontology.in_edges(term)])
                   
        for ancestor_Set in range(len(ancestor_storage)):
            if converged[ancestor_Set] == 1:
                newSet = set()
                ## check for acceptable new generalized terms
                for term in list_of_dicts[ancestor_Set].keys():
                    term_used_in_generalization_step = False
                    for candidate in list_of_dicts[ancestor_Set][term]:
                        desc_of_candidate = nx.descendants(ontology,candidate)

                        ## check for intersection
                        numberOfTerms = 0
                        intersectionCount = 0
                        for setTwo in range(len(ancestor_storage)):
                            if ancestor_Set != setTwo:
                                intersectionCount += len(list(set.intersection(desc_of_candidate, list_of_termsets[setTwo])))
                                numberOfTerms += len(list_of_termsets[setTwo])
                             
                        ## add the candidate to the set
                        if numberOfTerms == 0 or intersectionRatio >= float(intersectionCount) / float(numberOfTerms):
                            newSet.add(candidate)
                            term_used_in_generalization_step = True

                            
                    ## we keep the term if it hasn't been used
                    if not term_used_in_generalization_step:
                        newSet.add(term)
                        
                
                ## if we have generalized anything
                if ancestor_storage[ancestor_Set].difference(newSet):
                    ancestor_storage[ancestor_Set] = newSet
                    ## update iterations count
                    count_iterations_per_term[ancestor_Set] = update_dict_of_generalization_depth(count_iterations_per_term[ancestor_Set], newSet, count_iterations)
                else:               
                    ancestor_storage[ancestor_Set] = tmp_ancestor_storage[ancestor_Set]            
                    converged[ancestor_Set] = 0
            
        count_iterations += 1
        
    return((ancestor_storage, count_iterations_per_term))


def result_printing(class_names, subsets, evaluation):
    """
    A method which prints class names with the generalized set of terms
    :param class_names: list of class names
    :param subsets: a tuple of a list of generalized term sets and number of generalization iterations
    """    
    
   
    
    for i in range(len(class_names)):
        ## average generalization iterations through each element
        normalization = 0
        counter = 0
        class_dict = subsets[1][i]
        for item_key in class_dict.keys():
            normalization += class_dict[item_key]
            counter += 1
        if counter == 0:
            normalization = 0
        else:    
            normalization /= counter
        print("RESULT_TAG" + "\t" + str(class_names[i]) + "\t" + str(normalization) + "\t" + str(evaluation[i]) + "\t")

def generate_output_json(class_names, subsets, depth, connectedness):
    """
    A method which generates a simple output json suitable for further analysis.
    :param class_names: list of class names
    :param subsets: a tuple of a list of generalized term sets and number of generalization iterations
    :param depth: average generalization depth
    :param connectedness: average connectedness to other classes
    """
    
    outjson = {}
    for i in range(len(class_names)):
        class_name = str(class_names[i])
        subset = subsets[i]
        struct = {}
        struct['terms'] = list(subset)
        outjson[class_names[i]] = struct
    outjson['average_depth'] = depth
    outjson['average_association'] = connectedness
    return outjson
    
    
def generate_output_json_IC(class_names, subsets):
    """
    A method which generates a simple output json suitable for further analysis.
    :param class_names: list of class names
    :param subsets: a tuple of a list of generalized term sets and number of generalization iterations
    """
    
    outjson = {}
    for i in range(len(class_names)):
        class_name = str(class_names[i])
        subset = subsets[i]
        struct = {}
        struct['terms'] = list(subset)
        outjson[class_names[i]] = struct
    return outjson

def generate_output_json_without_depth(class_names, subsets, connectedness):
    """
    A method which generates a simple output json suitable for further analysis.
    :param class_names: list of class names
    :param subsets: a tuple of a list of generalized term sets and number of generalization iterations
    :param connectedness: average connectedness to other classes
    """

    outjson = {}
    for i in range(len(class_names)):
        class_name = str(class_names[i])
        subset = subsets[i]
        struct = {}
        struct['terms'] = subset
        outjson[class_names[i]] = struct
    outjson['average_association'] = connectedness
    return outjson

                
def evaluate(original, generalized):
    """
    A method which counts how many original terms were generalized in each set
    :param original: list of original term sets
    :param generalized: list of generalized term sets
    """
    result = []
    for i in range(len(original)):
        intersection = [value for value in generalized[i] if value in original[i]] 
        if len(original[i]) == 0:
            result.append(0)
        else:
            result.append((len(original[i]) - len(intersection)) / len(original[i]))
        
    
    return result
    
def generalization_depth(performance_dictionary):
    """
    A method which computes average generalization depth of terms
    :param performance_dictionary: list of dictionaries for each class which contain terms and their generalization depth
    """
    ##  average generalization depth
    generalization_average = 0
    counter = 0
    for i in range(len(performance_dictionary)):       
        class_dict = performance_dictionary[i]
        for item_key in class_dict.keys():
            generalization_average += class_dict[item_key]
            counter += 1
    if counter == 0:
        return -1
    return generalization_average / counter
    
    
def generalization_depth_ancestor(generalized, performance_dictionary):
    """
    A method which computes average generalization depth of terms for ancestor method
    :param performance_dictionary: dictionary which contain terms and their generalization depth
    :param generalized: list of sets of generalized results
    """
    ##  average generalization depth
    generalization_average = 0
    counter = 0
    
    for clas in range(len(generalized)):
        for term in generalized[clas]:
            counter += 1
            generalization_average += performance_dictionary[term]
    
    
    if counter == 0:
        return 0
    return generalization_average / counter
    
def class_connectedness(ontology, generalized, list_of_termsets):
    """
    A method which computes average term connectedness to other classes 
    :param ontology: ontology graph
    :param generalized: list of generalized term sets
    :param list_of_termsets: list of term sets per class
    """
    connectedness = 0
    counter = 0
    
    for i in range(len(generalized)):
        for term in generalized[i]:
            ##  if this is the starting term we will not count it's connectedness!
            basic_term = False
            for iter in range(len(list_of_termsets)):
                if term in list_of_termsets[iter]:
                    basic_term = True
                    break
            if not basic_term:
                descendants = nx.descendants(ontology,term)
                counter += 1
                for other_sets in range(len(list_of_termsets)):
                    if other_sets != i and len(set.intersection(descendants, list_of_termsets[other_sets])) > 0:
                        connectedness += 1

    if counter == 0:
        print(generalized)
        return 0
    return connectedness / counter

def extract_terms_from_explanations(explanations, attributes, gene_to_go_map, min_terms, step):

    """
    Given explanations, perform thesholding in order to get terms per class.
    :param explanations: Object containing SHAP-based explanations or similar.
    :param attributes: A vector of attributes.
    :param gene_to_go_map: file containing mapping from genes to GO terms
    :param min_terms: minimal number of terms taken for generalization per class
    :param step: multiplier for SHAP value threshold used to take most important terms of each class into generalization
    """
    
    term_sets_per_class = []
    class_names = []
    for class_name, explanation_vector in explanations.items():
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
        threshold = maxVector
        while True:
            if abs == 0:
                above_threshold = set(np.argwhere(greater_than_zero_vector >= threshold).flatten())
            else:
                above_threshold = set(np.argwhere(np.absolute(greater_than_zero_vector) >= threshold).flatten())
            if len(above_threshold) > min_terms or threshold < 0.1 * maxVector:
                break
            threshold *= step
        terms = set([x for enx,x in enumerate(attributes) if enx in above_threshold])
        all_terms = set()
        if gene_to_go_map:
            for term in terms:
                try:
                    mapped = gene_to_go_map[term]
                except:
                    try:
                        mapped = wn.synset(term + ".n.01").name()
                        #mapped = [x[1] for x in ontology.in_edges(term + ".n.01")][0]
                    except:
                        mapped = set()
                if isinstance(mapped, set):
                    all_terms = all_terms.union(mapped)
                else:
                    all_terms.add(mapped)

        else:
            all_terms = terms
        term_sets_per_class.append(all_terms)
        class_names.append(class_name)

    return term_sets_per_class, class_names


def generalize_selective_staircase(ontology_graph, explanations = None, attributes = None, target_relations = {"is_a","partOf"}, test_run = False, intersectionRatio = 0, abs = 0, print_results = False,gene_to_onto_map = None, min_terms = 5, step = 0.9):

    """
    A method which generalizes explanations based on the knowledge graph structure.
    :param ontology_graph: a NetworkX graph.fsampl
    :param explanations: a listdict with explanation data
    :param attributes: attribute names.
    :param target_relations: edge types in ontology used for generalization
    :param test_run: performing a test run
    :param intersectionRatio: maximum ratio of connected terms of other classes to a newly generalized term
    :param abs: 0 means absolute value is not used when determining terms with highest SHAP values, 1 means absolute value is used
    :param print_results: whether to print the taken terms before generalization
    :param gene_to_onto_map: file containing mapping from genes to GO terms
    :param min_terms: minimal number of terms taken for generalization per class
    :param step: multiplier for SHAP value threshold used to take most important terms of each class into generalization
    """
    term_sets_per_class, class_names = extract_terms_from_explanations(explanations,attributes, gene_to_onto_map, min_terms, step)
    
    subsets = selective_staircase_multiple_sets(term_sets_per_class, ontology_graph, intersectionRatio = intersectionRatio)

    
    #some basic evaluation of generalization
    evaluation = evaluate(term_sets_per_class, subsets[0])
    depth = generalization_depth(subsets[1])
    connected = class_connectedness(ontology_graph, subsets[0], term_sets_per_class)
    
    if print_results:  
        print("before generalization:" )
        for i in range(len(term_sets_per_class)):
            print("class: " + str(i) + str(list(term_sets_per_class[i])))
        result_printing(class_names, subsets, evaluation)

    return (generate_output_json(class_names, subsets[0], depth, connected), subsets[1])
    
def baseline_IC(ontology_graph, explanations = None, attributes = None, target_relations = {"is_a","partOf"}, test_run = False, intersectionRatio = 0, abs = 0, print_results = False,gene_to_onto_map = None, min_terms = 5, step = 0.9):
    """
    A method which generalizes explanations based on the knowledge graph structure.
    :param ontology_graph: a NetworkX graph.fsampl
    :param explanations: a listdict with explanation data
    :param attributes: attribute names.
    :param target_relations: edge types in ontology used for generalization
    :param test_run: performing a test run
    :param intersectionRatio: maximum ratio of connected terms of other classes to a newly generalized term
    :param abs: 0 means absolute value is not used when determining terms with highest SHAP values, 1 means absolute value is used
    :param print_results: whether to print the taken terms before generalization
    :param gene_to_onto_map: file containing mapping from genes to GO terms
    :param min_terms: minimal number of terms taken for generalization per class
    :param step: multiplier for SHAP value threshold used to take most important terms of each class into generalization
    """
    
    term_sets_per_class, class_names = extract_terms_from_explanations(explanations,attributes, gene_to_onto_map, min_terms, step)
    evaluation = evaluate(term_sets_per_class, term_sets_per_class)
    return generate_output_json_IC(class_names, term_sets_per_class)

def generalizeHedwig(ontology_graph, explanations = None, attributes = None, gene_to_onto_map = None):
    """
    A method which generalizes explanations based on the knowledge graph structure.
    :param ontology_graph: a NetworkX graph.fsampl
    :param explanations: a listdict with explanation data
    :param attributes: attribute names.
    :param gene_to_onto_map: file containing mapping from genes to GO terms
    """

    term_sets_per_class, class_names = extract_terms_from_explanations(explanations,attributes,gene_to_onto_map)
    partition = {}
    for enx, cx in enumerate(term_sets_per_class):
        for en in cx:
            partition[en] = enx
    generalized_termsets = get_hedwig_rules(partition)
    evaluation = evaluate(term_sets_per_class, generalized_termsets)
    if print_results:
        logging.info("before generalization:")
        for setIndex in range(len(term_sets_per_class)):
            print(class_names[setIndex] + ": " + str(list(term_sets_per_class[setIndex])))
        result_printing(class_names, subsets, evaluation)

    return generate_output_json(class_names, generalized_termsets)


def ancestor_multiple_sets(list_of_termsets, ontology, depthWeight):

    """
    :param list_of_termsets: termsets as found by explanations
    :param ontology: a nx graph
    :param depthWeight: higher weight gives greater importance to depth of generalization than the ration of intersection with terms of other classes. It doesnt delete old information!
    """
   
    converged = np.ones(len(list_of_termsets))
    tmp_ancestor_storage = None
    ancestor_storage = list_of_termsets.copy()
    combinedDepth = {}
    for i in range(len(list_of_termsets)):
        for term in list_of_termsets[i]:
            combinedDepth[term] = 0
    #combinedDepth = 0
    while not all(v == 0 for v in converged):
                       
        tmp_ancestor_storage = ancestor_storage.copy()
        for enx, termset in enumerate(ancestor_storage):
            list_of_this_termset = list(termset)
           
            if converged[enx] == 1:
                pairAncestorSet = set()
                setLength = len(list_of_this_termset)
                #boolean for whether the term was used to produce a pair ancestor or not
                used = [0] * setLength
                for item1 in range(setLength):
                    item2 = random.randint(0,setLength-1)
                    #for item2 in range(setLength):
                    if item1 != item2: 
                        #add ancestor of the pair of elements
                        ancestor_element = nx.lowest_common_ancestor(ontology, list_of_this_termset[item1], list_of_this_termset[item2])
                        if ancestor_element is not None:
                            #find just how much did we generalize and how much intersection there is with other classes
                            generalizationDepth = nx.shortest_path_length(ontology, ancestor_element, list_of_this_termset[item1])
                            depth2 = nx.shortest_path_length(ontology, ancestor_element, list_of_this_termset[item2])
                            # UP TO DISCUSSION - we are interested in the lowest of values
                            if depth2 < generalizationDepth:
                                generalizationDepth = depth2
                            #check intersection with other classes
                            descendants_of_val = nx.descendants(ontology,ancestor_element)  
                            intersectionCount = 0
                            numberOfTerms = 0
                            for setTwo in range(len(tmp_ancestor_storage)):
                                if enx != setTwo:
                                    intersectionCount+=len(set.intersection(descendants_of_val, list_of_termsets[setTwo]))
                                    numberOfTerms += len(list_of_termsets[setTwo])
                           
                            
                            # UP TO DISCUSSION - based on generalizationDepth and intersectionCount we somehow decide whether to include the element or not
                            intersectionRatio = intersectionCount/numberOfTerms
                            if generalizationDepth == -1 and intersectionRatio == 0 or generalizationDepth * depthWeight == 0 or intersectionRatio/(generalizationDepth * depthWeight) < 0.5:
                                pairAncestorSet.add(ancestor_element)
                                used[item1] = 1
                                used[item2] = 1
                                ## average depth + new depth
                                combinedDepth[ancestor_element] = (combinedDepth[list_of_this_termset[item1]] + combinedDepth[list_of_this_termset[item2]]) / 2 + generalizationDepth
                                #combinedDepth += generalizationDepth
                #pairAncestorSet.add(list_of_this_termset[e] for e in range(setLength) if used[e] == 0)
                for k in range(len(used)):
                    if used[k] == 0:
                        pairAncestorSet.add(list_of_this_termset[k])
                if len(pairAncestorSet) > 0:
                    ancestor_storage[enx] = pairAncestorSet
                                  

        for ancestor_Set in range(len(ancestor_storage)):         
            #newSet = ancestor_storage[ancestor_Set].difference(tmp_ancestor_storage[])            
            if  len(ancestor_storage[ancestor_Set]) <= 1 or ancestor_storage[ancestor_Set] == tmp_ancestor_storage[ancestor_Set]:
                #ancestor_storage[ancestor_Set] = newSet
                converged[ancestor_Set] = 0
            #else:                           
                #converged[ancestor_Set] = 0
             
    return((ancestor_storage, combinedDepth))
    
    
    
    
def generalize_ancestry(ontology_graph, explanations = None, attributes = None, target_relations = {"is_a","partOf"}, test_run = False, depthWeight = 0, abs = 0, print_results = False,gene_to_onto_map = None, min_terms = 5, step = 0.9):

    """
    A method which generalizes explanations based on the knowledge graph structure.
    :param ontology_graph: a NetworkX graph.fsampl
    :param explanations: a listdict with explanation data
    :param attributes: attribute names.
    :param target_relations: edge types in ontology used for generalization
    :param test_run: performing a test run
    :param depthWeight: higher weight gives greater importance to depth of generalization than the ration of intersection with terms of other classes 
    :param abs: 0 means absolute value is not used when determining terms with highest SHAP values, 1 means absolute value is used
    :param print_results: whether to print the taken terms before generalization
    :param gene_to_onto_map: file containing mapping from genes to GO terms
    :param min_terms: minimal number of terms taken for generalization per class
    :param step: multiplier for SHAP value threshold used to take most important terms of each class into generalization
    """
    
    term_sets_per_class, class_names = extract_terms_from_explanations(explanations,attributes, gene_to_onto_map, min_terms, step)
    
    subsets = ancestor_multiple_sets(term_sets_per_class, ontology_graph, depthWeight = depthWeight)

        
    evaluation = evaluate(term_sets_per_class, subsets[0])
    depth = generalization_depth_ancestor(subsets[0], subsets[1])
    connected = class_connectedness(ontology_graph, subsets[0], term_sets_per_class)
    
    if print_results:
        print("before generalization:" )
        for i in range(len(term_sets_per_class)):
            print(str(list(term_sets_per_class[i])))
        result_printing(class_names, subsets, evaluation)

    return generate_output_json(class_names, subsets[0], depth, connected)


def generalize_quick_ancestry(ontology_graph, explanations=None, attributes=None, target_relations={"is_a", "partOf"},
                       test_run=False, intersectionRatio=0, abs=0, print_results=False, gene_to_onto_map=None, min_terms=5,
                       step=0.9, iterations=20):
    """
    A method which generalizes explanations based on the knowledge graph structure.
    :param ontology_graph: a NetworkX graph.fsampl
    :param explanations: a listdict with explanation data
    :param attributes: attribute names.
    :param target_relations: edge types in ontology used for generalization
    :param test_run: performing a test run
    :param intersection_ratio: maximum ratio of connected terms of other classes to a newly generalized term
    :param abs: 0 means absolute value is not used when determining terms with highest SHAP values, 1 means absolute value is used
    :param print_results: whether to print the taken terms before generalization
    :param gene_to_onto_map: file containing mapping from genes to GO terms
    :param min_terms: minimal number of terms taken for generalization per class
    :param step: multiplier for SHAP value threshold used to take most important terms of each class into generalization
    """

    term_sets_per_class, class_names = extract_terms_from_explanations(explanations, attributes, gene_to_onto_map,
                                                                       min_terms, step)

    subsets = quick_ancestry_multiple_sets(term_sets_per_class, ontology_graph, intersection_ratio=intersectionRatio, iterations=iterations)

    evaluation = evaluate(term_sets_per_class, subsets)
    connected = class_connectedness(ontology_graph, subsets, term_sets_per_class)

    #if print_results:
     #   print("before generalization:")
      #  for i in range(len(term_sets_per_class)):
       #     print(str(list(term_sets_per_class[i])))
        #result_printing(class_names, subsets, evaluation)

    return generate_output_json_without_depth(class_names, subsets, connected)


def quick_ancestry_multiple_sets(list_of_termsets, ontology, intersection_ratio, iterations):
    """
    :param list_of_termsets: termsets as found by explanations
    :param ontology: a nx graph
    :param intersection_ratio: maximum ratio of connected terms of other classes to a newly generalized term
    """

    tmp_ancestor_storage = None
    ancestor_storage = list_of_termsets.copy()
    for a in range(iterations) :

        tmp_ancestor_storage = ancestor_storage.copy()
        for enx, termset in enumerate(ancestor_storage):
            list_of_this_termset = list(termset)

            
            pairAncestorSet = set()
            setLength = len(list_of_this_termset)
            # boolean for whether the term was used to produce a pair ancestor or not
            used = [0] * setLength
            for item1 in range(setLength):
                item2 = random.randint(0, setLength - 1)
              
                if item1 != item2:
                    # add ancestor of the pair of elements
                    ancestor_element = nx.lowest_common_ancestor(ontology, list_of_this_termset[item1],
                                                                 list_of_this_termset[item2])
                    if ancestor_element is not None:
                        # check intersection with other classes
                        descendants_of_val = nx.descendants(ontology, ancestor_element)
                        intersectionCount = 0
                        numberOfTerms = 0
                        for setTwo in range(len(tmp_ancestor_storage)):
                            if enx != setTwo:
                                intersectionCount += len(
                                    set.intersection(descendants_of_val, list_of_termsets[setTwo]))
                                numberOfTerms += len(list_of_termsets[setTwo])


                        if numberOfTerms == 0 or intersection_ratio >= float(intersectionCount) / float(numberOfTerms):
                            pairAncestorSet.add(ancestor_element)
                            used[item1] = 1
                            used[item2] = 1
                           
            for k in range(len(used)):
                if used[k] == 0:
                    pairAncestorSet.add(list_of_this_termset[k])
            if len(pairAncestorSet) > 0:
                ancestor_storage[enx] = pairAncestorSet

              

    return (ancestor_storage)