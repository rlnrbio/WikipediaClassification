# -*- coding: utf-8 -*-
"""
Created on Tue Sep 07 19:32:26 2021

@author: rapha
"""

from dataloading import load_combined_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from cleaning_utils import cleaning

import matplotlib.pyplot as plt
import scipy.sparse as sparse
import pickle
import numpy as np

random_seed = 42

settings_dict_sklearn =  {
    "decode_error": 'replace',
    "lowercase": True,
    "analyzer" : "word",
    "stop_words": "english",
    "max_df": 0.5, # remove words that appear in more than half of the texts
    "min_df": 10, # remove words that appear in less than 10 texts
    "max_features": None, # build vocabulary with a maximum number of features
    }


def vectorizeTfidf(savefile, vectorizer = "sklearn", save_features = True, sl = True):
    """

    Parameters
    ----------
    savefile : str
        Combined Sourcefolder of the wikidata to load, eg. wikidset_comb_1
    vectorizer : str, optional
        Which vectorizer to use. Options are "sklearn". The default is "sklearn".

    Returns
    -------
    None.

    """
    # load data to be vectorized
    texts, labels = load_combined_dataset(savefile=savefile)
    texts = [cleaning(t, sl) for t in texts]
    if sl: 
        title_tfidf = "sklearn_tfidfmat_clean.npz"
        title_features = "feature_names_clean.pickle"
    else:
        title_tfidf = "sklearn_tfidfmat.npz"
        title_features = "feature_names.pickle"
    
    # select vectorizer, currently only supports sklearn
    if vectorizer == "sklearn":
        vectorizer = TfidfVectorizer(**settings_dict_sklearn)
        X = vectorizer.fit_transform(texts)
        sparse.save_npz(savefile + "/" + title_tfidf, X)
        
    # save features for later feature importance analysis
    if save_features == True:
        feat_names = vectorizer.get_feature_names()
        with open(savefile + "/" + title_features, 'wb') as handle:
            pickle.dump(feat_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadTfidf(savefile, vectorizer = "sklearn", clean = True):
    # load tfidf values
    if vectorizer == "sklearn":    
        if clean:
            name = "/sklearn_tfidfmat_clean.npz"
        else:
            name = "/sklearn_tfidfmat.npz"

        mat = sparse.load_npz(savefile + name)
    return mat

def tfidf_dist(savefile, vectorizer = "sklearn", clean = True):
    # plot distribution of tfidf values
    mat = loadTfidf(savefile, vectorizer, clean)
    means = np.mean(mat, axis = 0)
    plt.hist(means)
    
        
def loadTfidfdataset(savefile, vectorizer = "sklearn", split = 0.2, clean = True):
    """
    

    Parameters
    ----------
    savefile : str
        Folder from which data should be loaded.
    vectorizer : str, optional
        Which vectorizer was used to create data. The default is "sklearn". Currently only "sklearn" is supported
    split : number between 0 and 1, optional
        Ratio of test to train data. The default is 0.2. Can also be none, then just one dataset is created
    clean : Bool, optional
        Select if lemmatized and stemmed data should be loaded, must have been created accordingly. The default is True.

    Returns
    -------
    TYPE
        Feature Matrix, labels.

    """
    with open(savefile + "/labels.pickle", "rb") as sf:
        labels = pickle.load(sf)
    if vectorizer == "sklearn":    
        if clean:
            name = "/sklearn_tfidfmat_clean.npz"
        else:
            name = "/sklearn_tfidfmat.npz"

        mat = sparse.load_npz(savefile + name)
    if split != None:
        texts_train, texts_test, labels_train, labels_test = train_test_split( 
            mat, labels, test_size=split, random_state=random_seed)
        return texts_train, texts_test, labels_train, labels_test
    else:
        return mat, labels

    
# Uncomment this to create the Tfidf Datasets: 
# vectorizeTfidf(savefile = "wikidset_comb_1_basic", vectorizer = "sklearn", cleaning = False)
# vectorizeTfidf(savefile = "wikidset_comb_2_basic", vectorizer = "sklearn")
# vectorizeTfidf(savefile = "wikidset_comb_5_basic", vectorizer = "sklearn")
# vectorizeTfidf(savefile = "wikidset_comb_10_basic", vectorizer = "sklearn")
