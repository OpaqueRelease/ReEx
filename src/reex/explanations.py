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
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
import time
import gzip
import networkx as nx
import obonet
import timeit
import sys
import os

try:
    import spyct
except:
    pass

from sklearn.feature_extraction.text import TfidfVectorizer

def get_instance_explanations(X, Y, subset = 1000, classifier_index = "gradient_boosting", explanation_method = "shap", shap_explainer = "kernel", text = False):
    
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
        #print(X_vectorized)
        X_vectorized = X_vectorized.todense()
        #print(X_vectorized)
        X = pd.DataFrame(X_vectorized)
        X.columns = vectorizer.get_feature_names()
        #X.columns = vectorizer.get_feature_names()
    logging.info("Feature pre-selection via Mutual Information ({}).".format(subset))
    #X = X.iloc[:,1:100]
    minf = mutual_info_classif(X.values, training_scores_encoded)
    top_k = np.argsort(minf)[::-1][0:subset]
    attribute_vector = X.columns[top_k]
    X = X.astype(float).values[:,top_k]
    skf = StratifiedKFold(n_splits=10)
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
        for train_index, test_index in skf.split(X, Y):
            enx+=1
            clf = model_dict[classifier_index]
            x_train = X[train_index]
            x_test = X[test_index]
            
            y_train = Y[train_index]
            y_test = Y[test_index]

            ## perform simple feature ranking
            minf = mutual_info_classif(x_train, y_train)
            top_k = np.argsort(minf)[::-1][0:subset]
            x_train = x_train[:,top_k]
            x_test = x_test[:,top_k]

            x_train = x_train.astype('float')
            y_train = y_train.astype('float')
            x_test = x_test.astype('float')
            y_test = y_test.astype('float')

            model = clf.fit(x_train, y_train)
            preds = model.predict(x_test)
            if len(np.unique(y_train)) > 1:
                average = "micro"
            perf = f1_score(preds,y_test, average = average)
            performances.append(perf)
            logging.info("Performance in fold {}, {} (F1)".format(enx, perf))
            ## different shap explainers
            if shap_explainer == "kernel":
                explainer = shap.KernelExplainer(model.predict_proba, x_train)
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
                cors_neg = np.array([enx for enx, pred_tuple in enumerate(zip(preds, y_test)) if pred_tuple[0] == pred_tuple[1] and pred_tuple[0] == unique_class])
                if cors_neg.size != 0:
                    shap_values = explainer.shap_values(x_test[cors_neg], nsamples = 10, verbose = False)
                    stack = np.mean(np.vstack(shap_values),axis = 0)
                    per_class_explanations[unique_class].append(stack)

        final_explanations = {}
        for class_name, explanation_set in per_class_explanations.items():
            final_explanations[class_name] = np.mean(np.matrix(explanation_set),axis = 0)
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


    return (final_explanations, attribute_vector)
