import shap
import transformers
import numpy as np
import scipy as sp
from transformers import AutoModel, AutoTokenizer,AutoModelForSequenceClassification
import torch
from nltk.wsd import lesk
import nltk
nltk.download('omw')

def get_explanations(data, labels):

    model = AutoModelForSequenceClassification.from_pretrained("IMSyPP/hate_speech_en")
    tokenizer = AutoTokenizer.from_pretrained("IMSyPP/hate_speech_en")

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

    for class_name in classes:
        per_class_max_shap_values[class_name] = {}
        per_class_explanations[class_name] = []
        class_subset = data.loc[labels == class_name]
        shap_values = explainer(class_subset, fixed_context=1)

        row_ix = 0
        for list_of_words in shap_values.data:
            word_ix = 0
            for word in list_of_words:
                syns = lesk(list_of_words, word)
                if syns is not None:
                    if syns.name() not in feature_names:
                        feature_names.append(syns.name())
                    if syns.name() not in per_class_max_shap_values[class_name] or per_class_max_shap_values[class_name][syns.name()] < shap_values.values[row_ix][word_ix]:
                        per_class_max_shap_values[class_name][syns.name()] = shap_values.values[row_ix][word_ix]
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
    return (per_class_explanations, feature_names)
