## This is the main blob of code responsible for generation of explanations

import shap
from sklearn.feature_selection import VarianceThreshold
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
from sklearn import svm
from sklearn.cluster import AffinityPropagation
import logging
from sklearn.cluster import MeanShift
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
import pickle
import time
from numpy import unique, where
import gzip
import networkx as nx
import timeit
import sys
import os
from nltk.corpus import wordnet
from nltk.wsd import lesk

try:
    import spyct
except:
    pass

from sklearn.feature_extraction.text import TfidfVectorizer

def fit_space(X, model_path="."):
    t2v_instance, tokenizer = load()
    features_matrix = []
    semantic_features = t2v_instance.transform(X)
    features_matrix.append(semantic_features)
    tfidf_words = tokenizer.transform(build_dataframe(X))
    features_matrix.append(tfidf_words)
    features = hstack(features_matrix)
    return features


def get_instance_explanations(X, Y, subset = 1000, classifier_index = "gradient_boosting", explanation_method = "shap", shap_explainer = "kernel", text = False, model_path=None, language='eng', clustering=False, feature_prunning=False):
    """
    A set of calls for obtaining aggregates of explanations.
    """
    ## label encoding
    #lab_enc = preprocessing.LabelEncoder()
    #training_scores_encoded = lab_enc.fit_transform(Y)
    # TODO: zakaj je potreben label encoder?

    training_scores_encoded = Y
    if text:
        vectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')
        X_vectorized = vectorizer.fit_transform(X)
        X_vectorized = X_vectorized.todense()
        X_usable = pd.DataFrame(X_vectorized)
        X_usable.columns = vectorizer.get_feature_names()
    else:
        X_usable = X.copy()
    if feature_prunning:
        logging.info("Feature pre-selection via Mutual Information ({}).".format(subset))
        minf = mutual_info_classif(X_usable.values, training_scores_encoded)
        top_k = np.argsort(minf)[::-1][0:subset]
        attribute_vector = X_usable.columns[top_k]
        X_usable = X_usable.astype(float).values[:,top_k]
    else:
        attribute_vector = X_usable.columns
        
    skf = StratifiedKFold(n_splits=3)
    performances = []
    enx = 0
    t_start = time.time()
    logging.info("Starting importance estimation ..  shape: {}".format(X.shape))

    per_class_explanations = defaultdict(list)
    classifier_mapping = ["gradient_boosting", "random_forest", "svm"]
    classifiers = [GradientBoostingClassifier(), RandomForestClassifier(n_estimators=10), svm.SVC(probability=True)] ## spyct.Model()

    model_dict = dict(zip(classifier_mapping, classifiers))
    
    if explanation_method == "shap":
        logging.info("Shapley-based explanations.")
        ## for the correctly predicted instances, remember shap values and compute the expected value at the end.
        for train_index, test_index in skf.split(X_usable, Y):
            pd.DataFrame(X.iloc[train_index]).to_csv("../results/train_split.csv")
            pd.DataFrame(X.iloc[test_index]).to_csv("../results/test_split.csv")
            enx+=1
            model = None
            clf = None
            if model_path:
                model = pickle.load(open(model_path, 'rb'))
            else:
                clf = model_dict[classifier_index]

            x_train = X_usable.iloc[train_index]
            x_test = X_usable.iloc[test_index]
            
            y_train = Y[train_index]
            y_test = Y[test_index]

            if not text:
                x_train = x_train.astype('float')
                #y_train = y_train.astype('float')
                x_test = x_test.astype('float')
                #y_test = y_test.astype('float')

            if not model_path:
                model = clf.fit(x_train, y_train)
            preds = model.predict(x_test)
            if len(np.unique(y_train)) > 1:
                average = "micro"
            perf = f1_score(preds,y_test, average = average)
            performances.append(perf)
            logging.info("Performance in fold {}, {} (F1)".format(enx, perf))
            ## different shap explainers
            if shap_explainer == "base":
                explainer = shap.Explainer(model.decision_function, x_train)
            if shap_explainer == "kernel":
                explainer = shap.KernelExplainer(model.decision_function, x_train)
            if shap_explainer == "tree":
                explainer = shap.TreeExplainer(model.predict_proba, x_train)
            if shap_explainer == "gradient":
                explainer = shap.GradientExplainer(model.predict_proba, x_train)
            if shap_explainer == "deep":
                explainer = shap.DeepExplainer(model.predict_proba, x_train)
            if shap_explainer == "sampling":
                explainer = shap.SamplingExplainer(model.predict_proba, x_train)
            if shap_explainer == "partition":
                explainer = shap.PartitionExplainer(model.predict_proba, x_train)

            for unique_class in set(preds):
                print("Class:", unique_class)
                cors_neg = np.array([enx for enx, pred_tuple in enumerate(zip(preds, y_test)) if pred_tuple[0] == pred_tuple[1] and pred_tuple[0] == unique_class and unique_class == "OFF"])
                print(cors_neg)
                if cors_neg.size != 0:
                    shap_values = explainer(x_test[cors_neg])
                    shap_values.feature_names = list(attribute_vector)
                    #shap.plots.bar(shap_values, max_display=20) 
                    #shap.summary_plot(shap_values, feature_names=list(attribute_vector), max_display=20)
                    #stack = np.mean(np.vstack(shap_values),axis = 0)
                    values_array = np.array(shap_values.values)
                    # for vector in range(len(values_array)):
                    #     for value in range(len(values_array[vector])):
                    #         if values_array[vector][value] < 0:
                    #             values_array[vector][value] = 0


                    ## CLUSTERING
                    if clustering:
                        #model = AffinityPropagation(damping=0.9)
                        #model.fit(values_array)
                        #yhat = model.predict(values_array)
                        model = MeanShift()
                        yhat = model.fit_predict(values_array)
                        clusters = unique(yhat)
                        print(clusters)
                        for cluster in clusters:
                            row_ix = where(yhat == cluster)
                            values_of_cluster = values_array[row_ix]
                            cluster_name = str(unique_class) + str(cluster)

                            cohorts = {"": values_of_cluster}
                            cohort_labels = list(cohorts.keys())
                            cohort_exps = list(cohorts.values())
                            for i in range(len(cohort_exps)):
                                if len(cohort_exps[i].shape) == 2:
                                    cohort_exps[i] = cohort_exps[i].mean(0)
                            features = cohort_exps[0].data
                            values = np.array([cohort_exps[i] for i in range(len(cohort_exps))])
                            per_class_explanations[cluster_name].append(values)
                    else:
                        per_class_explanations[unique_class].append(values_array)

            break # one train / test split

        final_explanations = {}
        for class_name, explanation_set in per_class_explanations.items():
            #final_explanations[str(class_name)] = np.mean(np.matrix(explanation_set),axis = 0).flatten()
            final_explanations[str(class_name)] = explanation_set
        average_perf = (np.mean(performances), np.std(performances))
        logging.info("Final performance: {}".format(average_perf))
 
    elif explanation_method == "class-ranking":
        logging.info("Ranking-based explanations.")
        unique_scores = np.unique(training_scores_encoded)
        final_explanations = {}
        for label in unique_scores:
            inx = np.where(training_scores_encoded == label)
            tx = VarianceThreshold().fit(X[inx]).variances_
            final_explanations[str(label)] = tx

    t_end = time.time() - t_start
    logging.info("Time spent on explanation estimation {}s.".format(t_end))

    disambiguated_feature_vector = []
    for feature in attribute_vector:
        ftr = feature.strip()
        disambiguated_feature = lesk([ftr], ftr, synsets=wordnet.synsets(ftr, lang=language))
        if disambiguated_feature is not None:
            disambiguated_feature_vector.append(disambiguated_feature.name())
        else:
            disambiguated_feature_vector.append("")

    return (final_explanations, disambiguated_feature_vector)
