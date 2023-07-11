## metrics that estimate generalization quality.
from misc import *
import numpy as np
import operator

def information_content(generalization, ontology, mappings):
    """
    A simple information content calculation
    : param generalization: resulting generalization object, needs "resulting_generalization" key
    : param ontology: a DAG
    : param mapping: mapping from core terms to an ontology

    IC = - \sum_k \log(p(term_k))

    """
    genes = set()
    counter = 0
    vsota = 0

    for k, v in mappings.items():
        genes.add(k)
        counter += 1
        vsota += len(v)
    print("Found " + str(len(genes)) + " genes mapped on average into " + str(vsota/counter) + "GO terms")

    mc = {}
    all_terms = set()
    for k,v in mappings.items():
        for el in v:
            all_terms.add(el)
            if el in mc:
                mc[el] += 1
            else:
                mc[el] = 1

    normalization = len(all_terms)
    logging.info("Found {} GO terms".format(normalization))
    class_ic = []
    vals = np.array(list(mc.values()))/normalization
    mterm = max(mc.items(), key=operator.itemgetter(1))
    minterm = min(mc.items(), key=operator.itemgetter(1))
    logging.info("Min IC: {}, Max IC = {}, freq min: {}, freq max = {}".format(-np.log(mterm[1]/normalization),-np.log(minterm[1]/normalization),mterm, minterm))

    for cname, explanations in generalization.items():
        if not isinstance(explanations, (float, int)):
            terms = explanations['terms']
            IC = 0
            ##  we shall not normalize with unscored terms (those not found in mappings file)
            unscored = 0
            for term in terms:
                if term in mc:                
                    p = mc[term]/normalization
                    IC+= (-np.log(p))
                else:
                    unscored += 1
            
            if len(terms) > unscored:
                IC /= len(terms) - unscored
                class_ic.append(IC)
    return np.mean(class_ic), np.max(class_ic), np.min(class_ic)

def calculate_average_distance(generalization, ontology):
    DG = list(nx.topological_sort(ontology))
    root = DG[0]

    number = 0
    sum = 0
    unscored = 0
    for cname, explanations in generalization.items():
        if not isinstance(explanations, (float, int)):
            terms = explanations['terms']
            for term in terms:
                root = DG[0]
                root_counter = 0
                number += 1
                if term in DG:
                    while True:
                        try:
                            sum += nx.shortest_path_length(ontology,root,term)
                            break
                        except:
                            root_counter += 1
                            root = DG[root_counter]
                else:
                    unscored += 1
    if number - unscored > 0:
        normalize = sum/(number- unscored)
    else:
        return 0

    return normalize


def compute_all_scores(generalization, ontology, mapping):
    """
    Traverse individual scores.
    """
    out_scores = {}
    mappings = read_generic_gaf(mapping)    
    IC = information_content(generalization, ontology, mappings)
    IC = [float(x) for x in IC] ## da ne cvili za tipi
    out_scores['IC'] = IC

    average_distance = calculate_average_distance(generalization, ontology)
    out_scores['root_distance'] = average_distance
    ## todo -> verjetno rabiva tudi lift, wracc in kaj podobnega.

    
    return out_scores
    

def compute_all_scores_text(generalization, ontology, mapping):
    """
    Traverse individual scores.
    """
    out_scores = {}

    average_distance = calculate_average_distance(generalization, ontology)
    out_scores['root_distance'] = average_distance
    ## todo -> verjetno rabiva tudi lift, wracc in kaj podobnega.

    
    return out_scores
    

    

            
            
    
    
        

