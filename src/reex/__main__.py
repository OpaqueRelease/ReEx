from misc import * ## parsers
from reasoning import * ## reasoners
from explanations import * ## explainers
from metrics import *
import argparse
import uuid
import json
import os
import os.path
from os import path 

parser = argparse.ArgumentParser()    
parser.add_argument('--expression_dataset',default='../example/data/Breast_A.csv', type = str)
parser.add_argument('--background_knowledge',default='../example/ontology/go-basic.obo', type = str)
## http://purl.obolibrary.org/obo/go/go-basic.obo -> whole ontology
## http://current.geneontology.org/ontology/subsets/goslim_agr.obo -> goSlim
parser.add_argument('--mapping_file',default='../example/mapping/goa_human.gaf.gz', type = str)
parser.add_argument('--intersection_ratio',default=0.2, type = float)
parser.add_argument('--depth_weight',default=0.1, type = float)
parser.add_argument('--subset_size', default=1000, type = int)
parser.add_argument('--classifier', default='gradient_boosting', type = str)
parser.add_argument('--absolute', action='store_true')
parser.add_argument('--explanation_method',default='class-ranking', type = str)
parser.add_argument('--reasoner',default='selective_staircase', type = str)
parser.add_argument('--min_terms',default=5, type = int)
parser.add_argument('--step',default=0.9, type = float)
#parser.add_argument('--baseline_IC',default=True, type = bool)
#parser.add_argument('--reverse_graph', action='store_false', default=True)
parser.add_argument('--results',default=False, type = bool)
parser.add_argument('--reverse_graph',default="true", type = str)
parser.add_argument('--baseline_IC', action='store_true')
parser.add_argument('--iterations',default=2, type = int)
parser.add_argument('--results_path',default='../results', type = str)
parser.add_argument('--SHAP_explainer',default='kernel', type = str)
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--text_input', action='store_true')
# ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/
args = parser.parse_args()

## unique id
salt = uuid.uuid4()
hash_value = hash(salt)
path_to_results = args.results_path
if not path.exists(path_to_results):
    os.mkdir(path_to_results)

#reversing
reversing = True
if args.reverse_graph == "false":
    reversing = False


## parse the background knowledge
if args.text_input:
    ontology_graph = get_ontology_text()
else :
    ontology_graph = get_ontology(obo_link = args.background_knowledge, reverse_graph = reversing)


## read the dataset
if args.text_input:
    parsed_dataset, target_vector, gene_to_onto_map = read_textual_dataset(args.expression_dataset)

    
else:
    parsed_dataset, target_vector, gene_to_onto_map = read_the_dataset(args.expression_dataset, attribute_mapping = args.mapping_file)

## generate explanations by using 10fcv
explanations, attributes = get_instance_explanations(parsed_dataset, target_vector, subset = args.subset_size, classifier_index = args.classifier, explanation_method = args.explanation_method, shap_explainer = args.SHAP_explainer, text = args.text_input)
if args.text_input:
    gene_to_onto_map = text_mapping(attributes, ontology_graph)
final_json = {'id' : hash_value,
              'reasoner':args.reasoner,
              'dataset':args.expression_dataset,
              'explanation_method':args.explanation_method,
              'absolute': args.absolute,
              'BK':args.background_knowledge,
              'subset_size':args.subset_size,
              'reverse_graph' : args.reverse_graph,
              'classifier':args.classifier,
              'min_terms':args.min_terms,
              'step':args.step}


outfile = open(path_to_results+'/'+str(hash_value)+'.json', 'w')

## reason and output
if args.reasoner == 'selective_staircase':
    (outjson, performance_dictionary) = generalize_selective_staircase(ontology_graph, explanations = explanations, attributes = attributes, test_run = False,  abs = args.absolute, intersectionRatio = args.intersection_ratio, gene_to_onto_map = gene_to_onto_map, print_results = args.results)
    if not args.text_input:
        scores = compute_all_scores(outjson, ontology_graph, args.mapping_file)
        final_json['scores'] = scores

    final_json['intersection_ratio'] = args.intersection_ratio
    final_json['resulting_generalization'] = outjson


elif args.reasoner == 'hedwig':
    outjson = generalizeHedwig(ontology_graph, explanations = explanations, attributes = attributes,gene_to_onto_map = gene_to_onto_map)
    final_json['intersection_ratio'] = args.intersection_ratio
    #final_json['scores'] = scores
    final_json['resulting_generalization'] = outjson
    
elif args.reasoner == 'ancestry':
    (outjson, performance_dictionary) = generalize_ancestry(ontology_graph, explanations = explanations, attributes = attributes, test_run = False,  abs = args.absolute, depthWeight = args.depth_weight, gene_to_onto_map = gene_to_onto_map, print_results = args.results)
    if not args.text_input:
        scores = compute_all_scores(outjson, ontology_graph, args.mapping_file)
        final_json['scores'] = scores
    final_json['depth_weight'] = args.depth_weight
    final_json['resulting_generalization'] = outjson

elif args.reasoner == 'quick_ancestry':
    outjson = generalize_quick_ancestry(ontology_graph, explanations=explanations, attributes=attributes, test_run=False,
                                 abs=args.absolute, intersectionRatio=args.intersection_ratio, gene_to_onto_map=gene_to_onto_map,
                                 print_results=args.results, iterations=args.iterations)
    if not args.text_input:
        scores = compute_all_scores(outjson, ontology_graph, args.mapping_file)
        final_json['scores'] = scores
    final_json['intersection_ratio'] = args.intersection_ratio
    final_json['resulting_generalization'] = outjson
    
    
if args.visualize:
    visualize_sets_of_terms(final_json, ontology_graph, performance_dictionary, target_vector)

dumper = json.dumps(final_json)
json.dump(dumper, outfile)

if not args.text_input:
    textualize_top_k_terms(final_json, args.mapping_file, args.background_knowledge, target_vector)

final_json = {'id':hash_value,
              'dataset':args.expression_dataset,
              'explanation_method':args.explanation_method,
              'absolute': args.absolute,
              'BK':args.background_knowledge,
              'subset_size':args.subset_size,
              'classifier':args.classifier,
              'min_terms':args.min_terms,
              'step':args.step}


##  baseline IC
if (args.baseline_IC):
    outfile_baseline = open(path_to_results+'/'+str(hash_value)+'_baseline.json', 'w')
    #print('baseline IC:')
    outjson = baseline_IC(ontology_graph, explanations = explanations, attributes = attributes, test_run = False,  abs = args.absolute, intersectionRatio = args.intersection_ratio, gene_to_onto_map = gene_to_onto_map)
    scores = compute_all_scores(outjson, ontology_graph, args.mapping_file)
    final_json['scores'] = scores
    final_json['resulting_generalization'] = outjson
    dumper = json.dumps(final_json)
    json.dump(dumper, outfile_baseline)
    outfile_baseline.close()
    print(final_json)
        
outfile.close()


