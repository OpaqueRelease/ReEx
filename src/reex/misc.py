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
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from textblob import Word
#from networkx.drawing.nx_pydot import graphviz_layout
#from networkx.drawing.nx_agraph import write_dot, graphviz_layout


def read_generic_gaf(gaf_file):
    """
    Reads .gaf mapping file
    """
    symmap = defaultdict(set)
    with gzip.open(gaf_file,"rt") as gf:
        for line in gf:
            line = line.strip().split("\t")
            if "UniProt" in line[0] and len(line) > 5:
                symmap[line[2]].add(line[4])
    logging.info("Found {} mappings.".format(len(symmap)))
    return symmap

def text_mapping(attributes):
    """
    Creates mapping dictionary of words and Wordnet terms
    """

    mapping = {}
    for col in attributes:
        try:
            #syns = wn.synsets(col)
            #mappedColumn = syns[0].name()
            #mapping[col] = mappedColumn
            if col is not None:
                mapping[col] = col  


            #mappedColumn = [x[1] for x in ontology.in_edges(col + ".n.01")][0]
            #mappedColumn = wn.synset(col + ".n.01").name()
            #mappedColumn = ontology.node(wn.synset().name())
            #mappedColumn = col + ".n.01"
        except:
            print("failed on: " + str(col))
    print(mapping)
    return mapping

def read_textual_dataset(dataset):
    """
    Reads a textual dataset
    """
    df = pd.read_csv(dataset, sep='\t', header=0)
    print(df)
    return df['data'], df['label'].values, None

def read_the_dataset(dataset_name, attribute_mapping = None):
    """
    Reads a nontextual dataset
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
            nmx = list(gaf_map[x])
            col_names.append(nmx[0])
    new_dx = rd.iloc[:,col_indices]
    logging.info("Considering DF of shape {}".format(new_dx.shape))
    return new_dx, target_vector, gaf_map
    
    
def get_ontology(obo_link = '../ontologies/go-basic.obo', reverse_graph = "false"):
    """
        Loads ontology for non-textual datasets.
    """
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

def recurse_custom(G, word):
    print("Entering with word: " + str(word))
    syns = wn.synsets(word)
    print(syns)
    w = syns[0]
    if not G.has_node(w.name()):
        G.add_node(w.name())
        for h in w.hypernyms():
            if h.name() != w.name():
                print (h)
                G.add_node(h.name())
                G.add_edge(w.name(),h.name())
                G = recurse_custom(G, h.name()[:-5])

    return G

def get_ontology_text_custom(mapping):
    nltk.download('wordnet')
    G = nx.DiGraph()

    for word in mapping.keys():
       node = wn.synset(mapping[word])
       temp_graph = closure_graph_fn(node, lambda s: s.hypernyms())
       G = nx.compose(G, temp_graph)

    print(nx.info(G))
    return G

def get_ontology_text():
    """
    Loads ontology for textual datasets
    """
    nltk.download('wordnet')
    G = nx.DiGraph()

    
    entity = wn.synset('entity.n.01')
    G = closure_graph_fn(entity, lambda s: s.hyponyms())
    print(nx.info(G))

    #print(set([x[0] for x in G.in_edges((wn.synset('meeting.n.01').name()))]))
    return G

def closure_graph_fn(synset, fn):
    """
    Constructs a NetworkX graph using nltk
    """
    seen = set()
    graph = nx.DiGraph()

    def recurse(s):
        if not s in seen:
            seen.add(s)
            graph.add_node(s.name())
            for s1 in fn(s):
                graph.add_node(s1.name())
                graph.add_edge(s1.name(), s.name())
                recurse(s1)

    recurse(synset)
    return graph


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

            draw_subgraph(set_of_top_k_terms, ontology, str(generalization_result), 2)
            counter += 1


def expand_set(set_of_terms, ontology, iterations):
    """
    Creates a set of descendats to the depth of **iterations
    """
    for i in range(iterations):
        new_terms = set()
        for term in set_of_terms:
            to_add =  set([x[1] for x in ontology.out_edges(term)])
            new_terms.update(to_add)
        set_of_terms.update(new_terms)
    return set_of_terms

def draw_subgraph(set_of_terms, ontology, class_name, depth):
    """
    Draws a graph
    """
    copy = set()
    copy.update(set_of_terms)
    combined_subgraph = expand_set(set_of_terms, ontology, depth)
    k = ontology.subgraph(combined_subgraph)
    color_map = []
    for node in k:
        if str(node) in copy:
            color_map.append('red')
        else:
            color_map.append('lightgrey')

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
        list_of_top_terms = []
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
                    print(("Class " + str(keyClass) + " :− " + str(id_to_name[term])).encode('utf8'))
                    first = False
                else:
                    print("^" + str(id_to_name[term]))
                genQ_dict[term] = -1
                list_of_top_terms.append(term)
        counter += 1


