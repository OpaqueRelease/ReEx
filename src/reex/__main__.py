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
from bert_shap import *

parser = argparse.ArgumentParser()    
parser.add_argument('--expression_dataset',default='../example/data/TCGA.csv', type = str)
parser.add_argument('--background_knowledge',default='../example/ontology/go-basic.obo', type = str)
parser.add_argument('--mapping_file',default='../example/mapping/goa_human.gaf.gz', type = str)
parser.add_argument('--intersection_ratio',default=0.03, type = float)
parser.add_argument('--cluster_intersection_ratio',default=0.1, type = float)
parser.add_argument('--depth_weight',default=10, type = float)
parser.add_argument('--cluster_depth_weight',default=100, type = float)
parser.add_argument('--subset_size', default=1000, type = int)
parser.add_argument('--classifier', default='gradient_boosting', type = str)
parser.add_argument('--absolute', action='store_true')
parser.add_argument('--explanation_method',default='shap', type = str)
parser.add_argument('--reasoner',default='selective_staircase', type = str)
parser.add_argument('--min_terms',default=3, type = int)
parser.add_argument('--step',default=0.7, type = float)
parser.add_argument('--results',default=False, type = bool)
parser.add_argument('--reverse_graph',default="true", type = str)
parser.add_argument('--baseline_IC', action='store_true')
parser.add_argument('--iterations',default=2, type = int)
parser.add_argument('--ancestors_searched',default=50, type = int)
parser.add_argument('--SHAP_explainer',default='base', type = str)
parser.add_argument('--text_input', action='store_true')
parser.add_argument('--wordnet', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--clustering', action='store_true')
parser.add_argument('--model',default=None, type = str)
parser.add_argument('--results_path',default='../results', type = str)
parser.add_argument('--bert', action='store_true')
parser.add_argument('--averaged', action='store_true')
parser.add_argument('--prune', action='store_true')
parser.add_argument('--disambiguate', action='store_true')
parser.add_argument('--lang',default='eng', type = str)
parser.add_argument('--twoclasses', action='store_true')
parser.add_argument('--static', default=0.0, type = float)
parser.add_argument('--plugin',default='', type = str)
parser.add_argument('--skip',action='store_true')



args = parser.parse_args()

path_to_results = args.results_path
if not path.exists(path_to_results):
    os.mkdir(path_to_results)

## unique id
salt = uuid.uuid4()
hash_value = hash(salt)


twoclasses = False
skip= False
if args.twoclasses:
    twoclasses = True
explanations = None
attributes = None
gene_to_onto_map = None
plugin_data = None

#reversing
reversing = True
if args.reverse_graph == "false":
    reversing = False

if args.plugin == '': # Continuation from json result
    ## read the dataset
    if args.text_input:
        parsed_dataset, target_vector, gene_to_onto_map = read_textual_dataset(args.expression_dataset)

        
    else:
        parsed_dataset, target_vector, gene_to_onto_map = read_the_dataset(args.expression_dataset, attribute_mapping = args.mapping_file)

    if args.bert:
        explanations, attributes = get_explanations(parsed_dataset, target_vector, args.averaged, args.lang)
    else:
        explanations, attributes = get_instance_explanations(parsed_dataset, target_vector, subset = args.subset_size, classifier_index = args.classifier, explanation_method = args.explanation_method, shap_explainer = args.SHAP_explainer, text = args.text_input, model_path=args.model, clustering=args.clustering, feature_prunning=args.prune, disambiguation=args.disambiguate, twoclasses=twoclasses)

else:
    plugin_data, gene_to_onto_map = get_plugin_data(args.plugin) # (terms_per_class, class_names)
# if args.text_input:
#     gene_to_onto_map = text_mapping(attributes)
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

## parse the background knowledge
if args.wordnet:
    ontology_graph = get_ontology_text_custom(attributes)
else :
    ontology_graph = get_ontology(obo_link = args.background_knowledge, reverse_graph = reversing)


## reason and output
if args.reasoner == 'selective_staircase':
    (outjson, performance_dictionary, baseline_terms, baseline_names) = generalize_selective_staircase(ontology_graph, explanations = explanations, attributes = attributes, test_run = False,  abs = args.absolute, intersectionRatio = args.intersection_ratio, gene_to_onto_map = gene_to_onto_map, print_results = args.results, min_terms=args.min_terms, cluster_intersection_ratio=args.cluster_intersection_ratio, static_threshold=args.static, plugin=plugin_data)
    if not args.text_input:
        pass
        #scores = compute_all_scores(outjson, ontology_graph, args.mapping_file)
        #final_json['scores'] = scores
    else:
        pass
        #scores = compute_all_scores_text(outjson, ontology_graph, args.mapping_file)
        #final_json['scores'] = scores

    final_json['intersection_ratio'] = args.intersection_ratio
    final_json['cluster_intersection_ratio'] = args.cluster_intersection_ratio
    final_json['resulting_generalization'] = outjson

    
elif args.reasoner == 'ancestry':
    (outjson, performance_dictionary, baseline_terms, baseline_names) = generalize_ancestry(ontology_graph, explanations = explanations, attributes = attributes, test_run = False,  abs = args.absolute, depthWeight = args.depth_weight, gene_to_onto_map = gene_to_onto_map, print_results = args.results, min_terms=args.min_terms, cluster_depth_weight=args.cluster_depth_weight, ancestors_searched=args.ancestors_searched, static_threshold=args.static, plugin=plugin_data)
    if not args.text_input:
        pass
        #scores = compute_all_scores(outjson, ontology_graph, args.mapping_file)
        #final_json['scores'] = scores
    else:
        pass
        #scores = compute_all_scores_text(outjson, ontology_graph, args.mapping_file)
        #final_json['scores'] = scores
    final_json['depth_weight'] = args.depth_weight
    final_json['cluster_depth_weight'] = args.cluster_depth_weight
    final_json['resulting_generalization'] = outjson


print("Generalization complete.")

if args.visualize:
    visualize_sets_of_terms(final_json, ontology_graph, performance_dictionary, target_vector)

outfile = open(path_to_results+'/'+str(hash_value)+'.json', 'w')
dumper = json.dumps(final_json)
json.dump(dumper, outfile)
print("JSON result saved.")

#if not args.text_input:
#   textualize_top_k_terms(final_json, args.mapping_file, args.background_knowledge, target_vector)

#print(final_json)

final_json = {'id':hash_value,
              'dataset':args.expression_dataset,
              'explanation_method':args.explanation_method,
              'absolute': args.absolute,
              'BK':args.background_knowledge,
              'subset_size':args.subset_size,
              'classifier':args.classifier,
              'min_terms':args.min_terms,
              'step':args.step}

        
outfile.close()


