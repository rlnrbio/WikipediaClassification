# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 16:26:33 2021

@author: rapha
"""
# In depth analysis of logistic Regression classification

from textvectorization import loadTfidfdataset
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



random_seed = 42

def run_logRegression(savefile, vectorizer, ttsplit,
                      getFeatImportance = True, clean = True):
    """
    

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
    # load data
    texts_train, texts_test, labels_train, labels_test = loadTfidfdataset(
        savefile, vectorizer = vectorizer, split = ttsplit, clean = clean)
    # create classifier and create predictions and calculate score
    classifier = LogisticRegression(random_state = random_seed).fit(texts_train, labels_train)
    preds = classifier.predict(texts_test)
    
    score = metrics.accuracy_score(labels_test, preds)
    report = metrics.classification_report(labels_test, preds)
    print(score)
    # save data with right labels and create feature importance df
    if clean == True:
        with open(savefile + "/" + vectorizer + "_tfidf_logReg_report_clean.txt", "w" ) as info:
            info.write(report)
        if getFeatImportance == True:
            with open(savefile + "/feature_names_clean.pickle", 'rb') as handle:
                features = pickle.load(handle)
            coefficients = classifier.coef_
            intercept = classifier.intercept_
            intercepts = list(intercept)*len(features)
            df = pd.DataFrame(data = (zip(features, list(coefficients[0]), intercepts)),
                                   columns = ["feature", "coefficient", "intercept"])
            df.to_csv(savefile + "/LogReg_featImportance_clean.csv")
            return score, df
        
        else:
            return score
    else:
        with open(savefile + "/" + vectorizer + "_tfidf_logReg_report.txt", "w" ) as info:
            info.write(report)
        if getFeatImportance == True:
            with open(savefile + "/feature_names.pickle", 'rb') as handle:
                features = pickle.load(handle)
            coefficients = classifier.coef_
            intercept = classifier.intercept_
            intercepts = list(intercept)*len(features)
            df = pd.DataFrame(data = (zip(features, list(coefficients[0]), intercepts)),
                                   columns = ["feature", "coefficient", "intercept"])
            df.to_csv(savefile + "/LogReg_featImportance.csv")
            return score, df
        else:
            return score


def feat_imp_plot_lr(featureImportance, savefile, plot_mostImportant = True, clean = True):
    # plot feature importance distribution from dataframe, optionally plot most important features (positive and negative)
    if clean: 
        clean_label = "clean"
    else:
        clean_label = ""
    fig, ax = plt.subplots(figsize= (5,10))
    fi = featureImportance["coefficient"]
    # count features above 1 and below -1
    aboveone = np.sum(fi>1)
    belowmone = np.sum(fi<-1)
    mean = np.mean(fi)
    sd = np.std(fi)
    var = np.var(fi)
    interc = featureImportance["intercept"][0]
    ax.boxplot(fi)
    ax.axhline(1, color = "green")
    ax.axhline(-1, color = "red")
    with open(savefile + "/LogReg_featImp_stats_" + clean_label + ".txt", "w" ) as info:
        info.write(savefile + "\nFeatsAboveOne: {} \nFeatsBelowMinusOne: {} \nMean: {} \n\
StdDev {} \nVar: {} \nInterc: {} ".format(aboveone, belowmone, 
                   mean, sd, var, interc))
    plt.savefig(savefile + "/LogReg_featImpdistribution_" + clean_label + ".png")
    
    if plot_mostImportant:
        # create most important feaures and save most important features as csv
        mifeatsplus = featureImportance[featureImportance["coefficient"] >1].sort_values(by = "coefficient", ascending = False)
        mifeatsminus = featureImportance[featureImportance["coefficient"] <-1].sort_values(by = "coefficient", ascending = True)
        mifeatsplus.to_csv(savefile + "/LogReg_mi_pos_feats_" + clean_label + ".csv")
        mifeatsminus.to_csv(savefile + "/LogReg_mi_neg_feats_" + clean_label + ".csv")

        
        fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize= (10,30))
        
        plusfeats = list(mifeatsplus["feature"])
        pluscoeff = list(mifeatsplus["coefficient"])
        
        minusfeats = list(mifeatsminus["feature"])
        minuscoeff = list(mifeatsminus["coefficient"])
        
        # Horizontal Bar Plot
        axs.barh(plusfeats, pluscoeff, color = "green")
         
        # Remove axes splines
        for s in ['top', 'bottom', 'left', 'right']:
            axs.spines[s].set_visible(False)
         
        # Remove x, y Ticks
        axs.xaxis.set_ticks_position('none')
        axs.yaxis.set_ticks_position('none')
         
        # Add padding between axes and labels
        axs.xaxis.set_tick_params(pad = 5)
        axs.yaxis.set_tick_params(pad = 10)
         
        # Add x, y gridlines
        axs.grid(b = True, color ='grey',
                linestyle ='-.', linewidth = 0.5,
                alpha = 0.2)
         
        # Show top values
        axs.invert_yaxis()
         
        # Add annotation to bars
        for i in axs.patches:
            plt.text(i.get_width() + 0.2, i.get_y()+0.5,
                      str(round((i.get_width()), 2)),
                      fontsize = 10, fontweight ='bold',
                      color ='black')
            
        fig.savefig(savefile + "/LogReg_pos_imp_feats_" + clean_label + ".png")
            
        fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize= (10,30))
        # Horizontal Bar Plot
        axs.barh(minusfeats, minuscoeff, color = "red")
         
        # Remove axes splines
        for s in ['top', 'bottom', 'left', 'right']:
            axs.spines[s].set_visible(False)
         
        # Remove x, y Ticks
        axs.xaxis.set_ticks_position('none')
        axs.yaxis.set_ticks_position('none')
         
        # Add padding between axes and labels
        axs.xaxis.set_tick_params(pad = 5)
        axs.yaxis.set_tick_params(pad = 10)
         
        # Add x, y gridlines
        axs.grid(b = True, color ='grey',
                linestyle ='-.', linewidth = 0.5,
                alpha = 0.2)
         
        # Show top values
        axs.invert_yaxis()
         
        # Add annotation to bars
        for i in axs.patches:
            plt.text(0.2, i.get_y()+0.5,
                      str(round((i.get_width()), 2)),
                      fontsize = 10, fontweight ='bold',
                      color ='black')
            
        fig.savefig(savefile + "/LogReg_neg_imp_feats_" + clean_label + ".png")

    return 1



# Run for balanced dataset, cleaned (lemmatized and stemmed) and simply cleaned dataset
score, feat_imp = run_logRegression(savefile = "wikidset_comb_1_basic", vectorizer = "sklearn", 
                                    ttsplit = 0.2, clean = True)
feat_imp_plot_lr(feat_imp, savefile = "wikidset_comb_1_basic", plot_mostImportant = True, 
                 clean = True)

score, feat_imp = run_logRegression(savefile = "wikidset_comb_1_basic", vectorizer = "sklearn", 
                                    ttsplit = 0.2, clean = False)
feat_imp_plot_lr(feat_imp, savefile = "wikidset_comb_1_basic", plot_mostImportant = True, 
                 clean = False)

# run_logRegression(savefile = "wikidset_comb_2_basic", vectorizer = "sklearn", ttsplit = 0.2)
# run_logRegression(savefile = "wikidset_comb_5_basic", vectorizer = "sklearn", ttsplit = 0.2)
# run_logRegression(savefile = "wikidset_comb_10_basic", vectorizer = "sklearn", ttsplit = 0.2)

# run_randomForest(savefile = "wikidset_comb_1_basic", vectorizer = "sklearn", ttsplit = 0.2)
# run_randomForest(savefile = "wikidset_comb_2_basic", vectorizer = "sklearn", ttsplit = 0.2)
# run_randomForest(savefile = "wikidset_comb_5_basic", vectorizer = "sklearn", ttsplit = 0.2)
# run_randomForest(savefile = "wikidset_comb_10_basic", vectorizer = "sklearn", ttsplit = 0.2)
