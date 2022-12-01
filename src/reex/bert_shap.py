import shap
import transformers
import numpy as np
import scipy as sp
from transformers import AutoModel, AutoTokenizer,AutoModelForSequenceClassification, TextClassificationPipeline
import torch
import json
from nltk.wsd import lesk
import nltk
nltk.download('omw')

def get_correctly_classified_instances(pipe, data, labels): ## Return dict of lists: class_name -> correctly classified instances
    classification_dictionary = {}

    for instance_ix in range(len(data)):
        #preprocessed = tokenizer(data[instance_ix], truncation=True)
        outputs = pipe(data[instance_ix])
        print(outputs)
        output_label = outputs[0]['label']
        if output_label == labels[instance_ix]:
            if output_label not in classification_dictionary:
                classification_dictionary[output_label] = [data[instance_ix]]
            else:
                classification_dictionary[output_label].append(data[instance_ix])
        return classification_dictionary

def save_instance_shapleys(class_name, shapley_values):
    json_dict = {}
    for index in range(len(shapley_values.data)):
        json_dict[index] = [shapley_values.data[index].tolist(), shapley_values.values[index].tolist()]

    with open('../results/' + class_name + '_instance_shapley.json', 'w') as convert_file:
        convert_file.write(json.dumps(json_dict))

def get_explanations(data, labels):

    model = AutoModelForSequenceClassification.from_pretrained("IMSyPP/hate_speech_en")
    tokenizer = AutoTokenizer.from_pretrained("IMSyPP/hate_speech_en")
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    # build an explainer using a token masker

    # define a prediction function
    def f(x):
        tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=500, truncation=True) for v in x])
        outputs = model(tv)[0].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores[:,1]) # use one vs rest logit units
        return val

    explainer = shap.Explainer(f, tokenizer)
    classes = ['NOT', 'OFF']
    per_class_explanations = {}
    per_class_max_shap_values = {}
    feature_names = []

    print(data)
    print(labels)

    #classification_dictionary = get_correctly_classified_instances(pipe, data, labels)
    disambiguation_dictionary = {}

    for class_name in classes:
        per_class_max_shap_values[class_name] = {}
        per_class_explanations[class_name] = []
        class_subset = data.loc[labels == class_name] #classification_dictionary[class_name]   #
        shap_values = explainer(class_subset, fixed_context=1)
        save_instance_shapleys(class_name, shap_values)

        row_ix = 0
        for list_of_words in shap_values.data:
            word_ix = 0
            for word in list_of_words:
                syns = lesk(list_of_words, word)
                if syns is not None:
                    if syns.name() not in feature_names:
                        feature_names.append(syns.name())
                        disambiguation_dictionary[syns.name()] = word
                    if syns.name() not in per_class_max_shap_values[class_name] or per_class_max_shap_values[class_name][syns.name()] < shap_values.values[row_ix][word_ix]:
                        per_class_max_shap_values[class_name][syns.name()] = shap_values.values[row_ix][word_ix]
                        disambiguation_dictionary[syns.name()] = word
                else:
                    # feature_names.append(None)
                    pass
                word_ix+=1
            row_ix += 1
        # shap_dict = {}

        # for ix in range(len(shap_values.values)):
        #     shap_dict[str(ix)] = shap_values.values[ix]
    for feature in feature_names:
        for class_name in classes:
            if feature in per_class_max_shap_values[class_name]:
                per_class_explanations[class_name].append(per_class_max_shap_values[class_name][feature])
            else:
                per_class_explanations[class_name].append(0)
    # to numpy arrays
    for key in per_class_explanations:
         per_class_explanations[key] = np.array(per_class_explanations[key])
    print(per_class_explanations)
    print(feature_names)
    ## export original words and their disambiguations
    with open('../results/disambiguation.json', 'w') as convert_file:
     convert_file.write(json.dumps(disambiguation_dictionary))

    return (per_class_explanations, feature_names)
