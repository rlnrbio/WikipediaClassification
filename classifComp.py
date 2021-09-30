# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:43:21 2021

@author: rapha
# classify  wikipedia articles, adapted from code written for NLP-course
"""


from textvectorization import loadTfidfdataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


from pprint import pprint

def evaluate_pred(targets, pred):
    # create basic metric vector
    acc = metrics.balanced_accuracy_score(targets, pred)
    pre = metrics.precision_score(targets, pred, average='weighted')
    rec = metrics.recall_score(targets, pred, average='weighted')
    f1 = metrics.f1_score(targets, pred, average='weighted')
    return [acc, pre, rec, f1]

# select analysis parameters
random_seed = 42
savefile = "wikidset_comb_1_basic"
vectorizer = "sklearn"
ttsplit = 0.2

# load datasets
vectors_train, vectors_test, targets_train, targets_test = loadTfidfdataset(
        savefile, vectorizer = vectorizer, split = ttsplit)

# NAIVE Bayes
from sklearn.naive_bayes import MultinomialNB

bayes = MultinomialNB(alpha = .01)
bayes.fit(vectors_train, targets_train)
bayes_pred = bayes.predict(vectors_test)
bayes_eval = evaluate_pred(targets_test, bayes_pred)

# SVM 
from sklearn.svm import SVC
svclass = SVC(gamma = "auto")
svclass.fit(vectors_train, targets_train)
sv_class_pred = svclass.predict(vectors_test)
sv_class_eval = evaluate_pred(targets_test, sv_class_pred)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(vectors_train, targets_train)
rf_pred = rf.predict(vectors_test)
rf_eval = evaluate_pred(targets_test, rf_pred)

# ADABOOST
from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier()
adb.fit(vectors_train, targets_train)
adb_pred = adb.predict(vectors_test)
adb_eval = evaluate_pred(targets_test, adb_pred)

# knn classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(vectors_train, targets_train)
knn_pred = knn.predict(vectors_test)
knn_eval = evaluate_pred(targets_test, knn_pred)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = random_seed)
classifier.fit(vectors_train, targets_train)
classifier_pred = classifier.predict(vectors_test)
classifier_eval = evaluate_pred(targets_test, classifier_pred)

# give and plot results
results = np.array([
    bayes_eval, sv_class_eval, rf_eval, adb_eval, knn_eval, classifier_eval])

labels = ["bayes", "svm", "RF", "adaboost", "knn", "logReg"]

fig, axs = plt.subplots(2,2, figsize = (14,8))
axs[0,0].bar(labels, results[:,0])
axs[0,0].set_title("Balanced accuracy")
axs[0,1].bar(labels, results[:,1])
axs[0,1].set_title("Precision")
axs[1,0].bar(labels, results[:,2])
axs[1,0].set_title("Recall")
axs[1,1].bar(labels, results[:,3])
axs[1,1].set_title("F1 score")



