# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 18:13:16 2021

@author: rapha
"""
# In depth analysis of random forest classification

from textvectorization import loadTfidfdataset
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random_seed = 42
savefile = "wikidset_comb_1_basic"


def run_randomForest(savefile, vectorizer, ttsplit, 
                     getFeatImportance = True, clean = True):
    """
    run random forest, very similar to logistic regression

    Parameters
    ----------
    savefile : string
        Folder from which to load and where to save data.
    vectorizer : string
        only sklearn supported.
    ttsplit : number between 0 and 1
        choose train test split.
    getFeatImportance : Bool, optional
        Should feature importance be calculated and plotted. The default is True.
    clean : Bool, optional
        Use lemmatized and stemmed data. The default is True.

    Returns
    -------
    TYPE
        score, feature importance data frame (only if feature importance was selected).


    """
    texts_train, texts_test, labels_train, labels_test = loadTfidfdataset(
        savefile, vectorizer = vectorizer, split = ttsplit, clean = clean)
    classifier = RandomForestClassifier(random_state = random_seed).fit(texts_train, labels_train)
    preds = classifier.predict(texts_test)
    
    score = metrics.accuracy_score(labels_test, preds)
    report = metrics.classification_report(labels_test, preds)
    print(score)
    if clean == True:
        with open(savefile + "/" + vectorizer + "_tfidf_randFor_report_clean.txt", "w" ) as info:
            info.write(report)
        if getFeatImportance == True:
            with open(savefile + "/feature_names_clean.pickle", 'rb') as handle:
                features = pickle.load(handle)
            importances = classifier.feature_importances_
            std = np.std([
                tree.feature_importances_ for tree in classifier.estimators_], axis=0)        
            df = pd.DataFrame(data = (zip(features, list(importances), list(std))),
                                   columns = ["feature", "importance", "std"])
            df.to_csv(savefile + "/RandForest_featImportance_clean.csv")
            return score, df
        else:
            return score
    else:
        with open(savefile + "/" + vectorizer + "_tfidf_randFor_report.txt", "w" ) as info:
            info.write(report)
        if getFeatImportance == True:
            with open(savefile + "/feature_names.pickle", 'rb') as handle:
                features = pickle.load(handle)
            importances = classifier.feature_importances_
            std = np.std([
                tree.feature_importances_ for tree in classifier.estimators_], axis=0)        
            df = pd.DataFrame(data = (zip(features, list(importances), list(std))),
                                   columns = ["feature", "importance", "std"])
            df.to_csv(savefile + "/RandForest_featImportance.csv")
            return score, df
        else:
            return score


def feat_imp_plot_rf(featureImportance, savefile, plot_mostImportant = True, clean = True):
    # plot feature importance distribution from dataframe, optionally plot most important features with mean and standard deviation

    if clean: 
        clean_label = "clean"
    else:
        clean_label = ""
    fig, ax = plt.subplots(figsize= (5,10))
    fi = featureImportance["importance"]
    # select features above 0.001 and 0.0001
    above001 = np.sum(fi>0.001)
    above0001 = np.sum(fi>0.0001)
    mean = np.mean(fi)
    mean_sd = np.mean(featureImportance["std"])
    ax.boxplot(fi)
    ax.axhline(0.001, color = "green")
    ax.axhline(0.0001, color = "lightgreen")
    with open(savefile + "/RandForest_featImp_stats_" + clean_label + ".txt", "w" ) as info:
        info.write(savefile + "\nFeatsAbove0.001: {} \nFeatsAbove0.0001: {} \nMean: {} \n\
MeanStdDev {}".format(above001, above0001, mean, mean_sd))
    plt.savefig(savefile + "/RandForest_featImpdistribution_" + clean_label + ".png")
    
    if plot_mostImportant:
        # plot features that are above 0.001
        mifeatsplus001 = featureImportance[featureImportance["importance"] >0.001].sort_values(by = "importance", ascending = False)
        mifeatsplus0001 = featureImportance[featureImportance["importance"] >0.0001].sort_values(by = "importance", ascending = False)
        mifeatsplus001.to_csv(savefile + "/RandForest_mi_feats001_" + clean_label + ".csv")
        mifeatsplus0001.to_csv(savefile + "/RandForest_mi_feats0001_" + clean_label + ".csv")

        
        
        plusfeats = list(mifeatsplus001["feature"])
        pluscoeff = list(mifeatsplus001["importance"])
        plusstd = list(mifeatsplus001["std"])
        
        forest_importances = pd.Series(pluscoeff, index=plusfeats)
        
        fig, axs = plt.subplots(figsize= (10,8))

        #Use MDI since Feature Permutation is too costly in terms of computation 
        #to be used on so many features 
        forest_importances.plot.bar(yerr=plusstd, ax=axs, color = "grey")
        axs.set_title("Feature importances using MDI")
        axs.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        fig.savefig(savefile + "/RandForest_featImpMDI_" + clean_label + ".png")
        
        
    return 1


# Run for balanced dataset, cleaned (lemmatized and stemmed) and simply cleaned dataset
score, feat_imp = run_randomForest(savefile = "wikidset_comb_1_basic", vectorizer = "sklearn", 
                                   ttsplit = 0.2, clean = True)
feat_imp_plot_rf(feat_imp, savefile = "wikidset_comb_1_basic", plot_mostImportant = True,
                 clean = True)
score, feat_imp = run_randomForest(savefile = "wikidset_comb_1_basic", vectorizer = "sklearn", 
                                   ttsplit = 0.2, clean = False)
feat_imp_plot_rf(feat_imp, savefile = "wikidset_comb_1_basic", plot_mostImportant = True,
                 clean = False)
