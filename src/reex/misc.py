## parsers and such.
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
import pandas as pd
import time
import gzip
import networkx as nx
import obonet
import timeit
from collections import defaultdict
import sys
import os
import numpy as np
from matplotlib import pyplot as plt

def list_to_ancestor_set(graph, list):
    result = set()
    for iterator in list:
        result = result.union(nx.descendants(graph,iterator))
    return result

def read_generic_gaf(gaf_file):

    symmap = defaultdict(set)
    with gzip.open(gaf_file,"rt") as gf:
        for line in gf:
            line = line.strip().split("\t")
            if "UniProt" in line[0] and len(line) > 5:
                symmap[line[2]].add(line[4])
    logging.info("Found {} mappings.".format(len(symmap)))
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
            nmx = list(gaf_map[x])
            col_names.append(nmx[0])
    logging.info("Found {} GO maps.".format(len(col_indices)))
    new_dx = rd.iloc[:,col_indices]
#    new_dx.columns = col_names
    logging.info("Considering DF of shape {}".format(new_dx.shape))
    return new_dx, target_vector, gaf_map

def get_ontology(obo_link = '../ontologies/go-basic.obo', reverse_graph = "false"):

    try:
        graph = obonet.read_obo(obo_link)
    except Exception as es:
        logging.info(es)
        graph = obonet.read_obo(obo_link)
    logging.info(obo_link)
    numberOfNodes = graph.number_of_nodes() 
    
    logging.info("Number of nodes: {}".format(numberOfNodes))
    reverseGraph = nx.MultiDiGraph()

    ## generate whole graph first, we'll specialize later.
    wholeset = set()
    for edge in list(graph.edges()):
        edge_info = set(graph.get_edge_data(edge[0], edge[1]).keys())
        wholeset = wholeset.union(edge_info)
        for itype in edge_info:
            if itype == "is_a" or itype == "part_of":
                if reverse_graph == "true":
                    reverseGraph.add_edge(edge[1], edge[0], type = itype)
                else:
                    reverseGraph.add_edge(edge[0], edge[1], type = itype)
    logging.info(nx.info(reverseGraph))
    tnum = len(wholeset)
    logging.info("Found {} unique edge types, {}".format(tnum," | ".join(wholeset)))
    return reverseGraph
    
    
def get_ontology_ancestor(obo_link = '../ontologies/go-basic.obo', reverse_graph = "false"):

    try:
        graph = obonet.read_obo(obo_link)
    except Exception as es:
        logging.info(es)
        graph = obonet.read_obo(obo_link)
        #obo_link = 'http://purl.obolibrary.org/obo/go/go-basic.obo'

    logging.info(obo_link)
    numberOfNodes = graph.number_of_nodes() 
    
    logging.info("Number of nodes: {}".format(numberOfNodes))
    reverseGraph = nx.DiGraph()

    ## generate whole graph first, we'll specialize later.
    wholeset = set()
    for edge in list(graph.edges()):
        edge_info = set(graph.get_edge_data(edge[0], edge[1]).keys())
        wholeset = wholeset.union(edge_info)
        for itype in edge_info:
            if itype == "is_a" or itype == "part_of":
                if reverse_graph == "true":
                    reverseGraph.add_edge(edge[1], edge[0], type=itype)
                else:
                    reverseGraph.add_edge(edge[0], edge[1], type=itype)
    logging.info(nx.info(reverseGraph))
    tnum = len(wholeset)
    logging.info("Found {} unique edge types, {}".format(tnum," | ".join(wholeset)))
    return reverseGraph
    

def visualize_sets_of_terms(json, ontology, dict, class_names,  k = 3):
    """
        Find the most generalized terms for each class, and visualize the subgraph of this term with depth *k*
    """
    counter = 0
    for generalization_result in json['resulting_generalization'].keys():
        if generalization_result  !="average_depth" and generalization_result != "average_association":
            set1 = json['resulting_generalization'][generalization_result]["terms"]
            working_dict = dict[0]

            set_of_top_k_terms = set()

            for iter in range(k):
                if iter < len(set1):
                    max = 0
                    term = ""
                    for i in set1:
                        if i in working_dict.keys():
                            if working_dict[i] >= max:
                                max = working_dict[i]
                                term = i

                    set_of_top_k_terms.add(term)
                    working_dict[term] = -1

            draw_subgraph(set_of_top_k_terms, ontology, str(generalization_result))
            counter += 1


def expand_set(set_of_terms, ontology, iterations):
    for i in range(iterations):
        new_terms = set()
        for term in set_of_terms:
            to_add =  set([x[1] for x in ontology.out_edges(term)])
            new_terms.update(to_add)
        set_of_terms.update(new_terms)
    return set_of_terms

def draw_subgraph(set_of_terms, ontology, class_name):
    copy = set()
    copy.update(set_of_terms)
    combined_subgraph = expand_set(set_of_terms, ontology, 2)
    k = ontology.subgraph(combined_subgraph)
    color_map = []
    for node in k:
        if str(node) in copy:
            color_map.append('red')
        else:
            color_map.append('lightgrey')

    #pos = graphviz_layout(k, prog='dot')
    pos = nx.spring_layout(k)
    plt.title("Terms for class " + class_name)
    nx.draw(k, pos = pos, with_labels=True, node_color = color_map)
    plt.show()
    plt.clf()


def IC_of_a_term(term, mapping, mc, normalization):
    """
        Calculates IC of a term
    """

    IC = 0
    if term in mc:
        p = mc[term] / normalization
        IC += (-np.log(p))
    else:
        ## 1000 as in impossibly high IC
        IC = 1000

    return IC


def textualize_top_k_terms(json_data, mapping, obo_link, class_names,  k_number = 5):
    """
        This method prints the names of the *k_number* most important terms for each class (according to genQ)
    """
    try:
        graph = obonet.read_obo(obo_link)
    except Exception as es:
        logging.info(es)
        graph = obonet.read_obo(obo_link)

    id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}

    ## go through mapping
    mc = {}
    all_terms = set()
    mappings = read_generic_gaf(mapping)
    for k, v in mappings.items():
        for el in v:
            all_terms.add(el)
            if el in mc:
                mc[el] += 1
            else:
                mc[el] = 1
    normalization = len(all_terms)

    counter = 0
    for keyClass in json_data["resulting_generalization"].keys():
        first = True
        print()
        if keyClass != "average_depth" and keyClass != "average_association":
            genQ_dict = {}
            for term in json_data["resulting_generalization"][keyClass]["terms"]:
                IC = IC_of_a_term(term, mappings, mc, normalization)
                genQ = 1 - IC / 9.82
                genQ_dict[term] = genQ
            for n in range(k_number):
                max = 0
                term = ""
                for k,v in genQ_dict.items():
                    if v >= max:
                        max = v
                        term = k
                if first:
                    print("Class " + str(keyClass) + " :− " + str(id_to_name[term]))
                    first = False
                else:
                    print("^" + str(id_to_name[term]))
                genQ_dict[term] = -1
        counter += 1


