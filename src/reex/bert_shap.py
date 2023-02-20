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
from nltk.corpus import wordnet

class_name_mapping = {
    "LABEL_0": "NOT",
    "LABEL_1": "OFF",
    "LABEL_2": "OFF",
    "LABEL_3": "OFF"
}

def get_correctly_classified_instances(pipe, data, labels): ## Return dict of lists: class_name -> correctly classified instances
    classification_dictionary = {}

    for instance_ix in range(len(data)):
        #preprocessed = tokenizer(data[instance_ix], truncation=True)
        outputs = pipe(data[instance_ix])
        print(outputs)
        output_label = outputs[0]['label']
        if class_name_mapping[output_label] == labels[instance_ix]:
            if labels[instance_ix] not in classification_dictionary:
                classification_dictionary[labels[instance_ix]] = [data[instance_ix]]
            else:
                classification_dictionary[labels[instance_ix]].append(data[instance_ix])
    return classification_dictionary

def save_instance_shapleys(class_name, shapley_values):
    json_dict = {}
    for index in range(len(shapley_values.data)):
        json_dict[index] = [shapley_values.data[index].tolist(), shapley_values.values[index].tolist()]

    with open('../results/' + class_name + '_instance_shapley.json', 'w') as convert_file:
        convert_file.write(json.dumps(json_dict))

def get_explanations(data, labels, averaged):

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

    classification_dictionary = get_correctly_classified_instances(pipe, data, labels)
    print(classification_dictionary)
    disambiguation_dictionary = {}

    for class_name in classes:
        class_subset = data.loc[labels == class_name] #classification_dictionary[class_name]   #
        shap_values = explainer(class_subset, fixed_context=1)

        if averaged:
            shap.plots.bar(shap_values, max_display=20)
            save_instance_shapleys(class_name, shap_values)
            # print(shap_values.values)
            # per_class_explanations[class_name] = np.abs(shap_values.values).mean(0)
            # feature_names = shap_values.data

            # print(per_class_explanations)
            # print(feature_names)

            cohorts = {"": shap_values}
            cohort_labels = list(cohorts.keys())
            cohort_exps = list(cohorts.values())
            for i in range(len(cohort_exps)):
                if len(cohort_exps[i].shape) == 2:
                    cohort_exps[i] = cohort_exps[i].abs.mean(0)
            features = cohort_exps[0].data
            values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])

            print(values)
            print(sum(values))

            zeros = [0] * len(feature_names) # Features of other classes set to 0
            zeros.extend(list(sum(values)))
            per_class_explanations[class_name] = zeros
            syn_names = cohort_exps[0].feature_names
            lemmas = [lesk([item], item, synsets=wordnet.synsets(item, lang='eng')) for item in syn_names]
            print(lemmas)
            lemma_names = []
            for lemma in lemmas:
                if lemma is not None:
                    lemma_names.append(lemma.name())
                else:
                    lemma_names.append("")
            print(lemma_names)
            feature_names.extend(lemma_names)
            print(feature_names)

        else:
            per_class_explanations[class_name] = []
            per_class_max_shap_values[class_name] = {}  
            row_ix = 0
            for list_of_words in shap_values.data:
                word_ix = 0
                for word in list_of_words:
                    syns = lesk(list_of_words, word, synsets=wordnet.synsets(word, lang='eng'))
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
    if averaged:
        for key in per_class_explanations.keys():
            extend_with_zeros = [0] * (len(feature_names) - len(per_class_explanations[key]))
            per_class_explanations[key].extend(extend_with_zeros)
        return (per_class_explanations, feature_names)

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
