# -*- coding: utf-8 -*-
"""
Created on Fri Sep 03 14:40:45 2021

@author: rapha
"""
import pandas as pd
import numpy as np
import pickle
from operator import itemgetter
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

random_seed = 42

def load_metadata(res_folder = "wikiresults"):
    # load metadata from articles
    meta = pd.read_csv(res_folder + "/metadata.csv", index_col = None)
    return meta

def load_articles(selectedArticles, res_folder = "wikiresults"):
    # load article pickle files including good article tags
    texts = list()
    good_articles = list()
    
    uniqueFiles = np.unique(selectedArticles["InFile"])
    for fID in uniqueFiles:
        fIDstr = res_folder + "/wiki" + str(fID).zfill(2) + ".pickle"
        infile = selectedArticles[selectedArticles["InFile"] == fID]
        infileIDs = list(infile["ProjectID"])
        newgood = list(infile["goodArticle"])
        
        with open(fIDstr, 'rb') as handle:
            tdict = pickle.load(handle)
        tdict = dict(tdict)    
        selected = list(itemgetter(*infileIDs)(tdict))
        texts = texts + selected
        good_articles = good_articles + newgood
    
    return good_articles, texts
            
# create identical textlength distributions:
# Sample 
def accum(l, size):
    # helper function to accumulate a list of numbers, used for sampling metadata
    newlist = list()
    value = 0
    counter = 0
    for elem in l:
        if counter < size-1:
            value += elem
            counter +=1
        else:
            newlist.append(value)
            value, counter = 0, 0
    return newlist

def sample_metadata(meta, sizefactor = 10, max_sampling_size = 200000, sampling_bin_size = 100, loc_factor = 1, 
                    no_shorter = True, plot = False, random_seed = random_seed):
    # This functions samples the metadata according to a fitted Gamma distribution
    # create length 
    article_length_good = meta[meta["goodArticle"] == True]["textlength"]
    
    sampled_nogood = pd.DataFrame(data=None, columns=meta.columns)
    
    good_articles = meta[meta["goodArticle"] == True]
    nogood_articles = meta[meta["goodArticle"] == False]

    fit_alpha, fit_loc, fit_beta=stats.gamma.fit(article_length_good)
    fit_loc = loc_factor*fit_loc
    # create sampling bins of 1000:
    x = list(range(0, max_sampling_size +1))
    y = stats.gamma.pdf(x, a=fit_alpha, scale=fit_beta, loc = fit_loc)
    y_accum = accum(y, sampling_bin_size)
    x_upper_lim = np.array(x[::100][1:])
    
    n_sample = len(article_length_good)*sizefactor
    
    to_chose = (np.round(np.array(y_accum)*n_sample))
    
    if (no_shorter == True):
        to_keep = x_upper_lim >= np.min(article_length_good)
        to_chose = to_chose[to_keep]
        x_upper_lim = x_upper_lim[to_keep]
    
    for i in range(len(x_upper_lim)):
        ul = x_upper_lim[i]
        n = np.int(to_chose[i])
        ll = ul-sampling_bin_size
        select = nogood_articles.loc[(nogood_articles["textlength"] >= ll) & (nogood_articles["textlength"] < ul)]
        sampled = select.sample(n, replace = True, random_state = random_seed)
        sampled_nogood = sampled_nogood.append(sampled)
        
    if plot == True:
        article_length_nogood = sampled_nogood["textlength"]
        # plot histogram of article lengths:
        
        fig, ax = plt.subplots(figsize= (10,8))
        ax.hist(article_length_good, color = "green", bins = 80, alpha = 0.6)
        ax.axvline(np.mean(article_length_good), color = "green")
        
        # set x-axis label
        ax.set_xlabel("Article Size",fontsize=14)
        # set y-axis label
        ax.set_ylabel("# Good Articles", color = "green", fontsize=14)
        
        ax2=ax.twinx()
        ax2.hist(article_length_nogood, color = "red", bins = 80, alpha = 0.6)
        ax2.axvline(np.mean(article_length_nogood), color = "red")
        
        ax2.set_ylabel("# NoGood Articles", color = "red", fontsize = 14)
        
        return good_articles, sampled_nogood, fig
    else:
       return good_articles, sampled_nogood

def sample_meta_basic(meta, sizefactor = 10, max_sampling_size = 200000, sampling_bin_size = 1000,
                    no_shorter = True, plot = False, random_seed = random_seed):
    # Sample data based on binned article length frequency, this was chosen for the experiments
    # create length 
    article_length_good = meta[meta["goodArticle"] == True]["textlength"]
    
    sampled_nogood = pd.DataFrame(data=None, columns=meta.columns)
    
    good_articles = meta[meta["goodArticle"] == True]
    nogood_articles = meta[meta["goodArticle"] == False]

    # create sampling bins of 1000:
    x = list(range(0, max_sampling_size +1))
    
    
    x_upper_lim = np.array(x[::sampling_bin_size][1:])
    x_lower_lim = np.array(x[::sampling_bin_size][:-1])
    
    
    n_sample = len(article_length_good)*sizefactor
        
    if (no_shorter == True):
        to_keep = x_upper_lim >= np.min(article_length_good)
        x_upper_lim = x_upper_lim[to_keep]
        x_lower_lim = x_lower_lim[to_keep]
    
    for i in range(len(x_upper_lim)):
        ul = x_upper_lim[i]
        ll = x_lower_lim[i]     
        number_goods = sum((article_length_good >= ll) & (article_length_good < ul))
        n = np.int(number_goods*sizefactor)

        select = nogood_articles.loc[(nogood_articles["textlength"] >= ll) & (nogood_articles["textlength"] < ul)]
        sampled = select.sample(n, replace = True, random_state = random_seed)
        sampled_nogood = sampled_nogood.append(sampled)
        
    if plot == True:
        article_length_nogood = sampled_nogood["textlength"]
        # plot histogram of article lengths:
        
        fig, ax = plt.subplots(figsize= (10,8))
        ax.hist(article_length_good, color = "green", bins = 80, alpha = 0.6)
        ax.axvline(np.mean(article_length_good), color = "green")
        
        # set x-axis label
        ax.set_xlabel("Article Size",fontsize=14)
        # set y-axis label
        ax.set_ylabel("# Good Articles", color = "green", fontsize=14)
        
        ax2=ax.twinx()
        ax2.hist(article_length_nogood, color = "red", bins = 80, alpha = 0.6)
        ax2.axvline(np.mean(article_length_nogood), color = "red")
        
        ax2.set_ylabel("# NoGood Articles", color = "red", fontsize = 14)
        
        return good_articles, sampled_nogood, fig
    else:
       return good_articles, sampled_nogood



def create_dataset(meta, sizefactor, savefile, max_sampling_size = 150000, loc_factor = 1.1, 
                   test_size = 0.2, random_seed = random_seed, sample_type = "basic"):
    # Create test and train datasets already, not used for later application 
    # since combined datasets are needed for textvectorization
    # possible sample types are basic and beta
    if os.path.isdir(savefile):
        os.mkdir(savefile)
    # sample articles based on metadata
    if sample_type == "beta":
        g, ng, plot = sample_metadata(meta = meta, sizefactor = sizefactor, max_sampling_size = max_sampling_size, sampling_bin_size = 100, 
                                loc_factor = loc_factor, no_shorter = True, plot = True)
    if sample_type == "basic":
        g, ng, plot = sample_meta_basic(meta = meta, sizefactor = sizefactor, max_sampling_size = max_sampling_size, sampling_bin_size = 100, 
                                no_shorter = True, plot = True)
    else: 
        print("wrong sample type")
        return 1
    # save distribution plots from sampling functions
    plot.savefig(savefile + "/distribution_plot.png")
    # load only those articles that were sampled
    g_data = load_articles(g)
    ng_data = load_articles(ng)
    # create combined dataset
    labels = g_data[0] + ng_data[0]
    texts = g_data[1] + ng_data[1]
    # shuffle and traintestsplit
    labels, texts = shuffle(labels, texts, random_state=random_seed)
    texts_train, texts_test, labels_train, labels_test = train_test_split( texts, labels, test_size=test_size, random_state=random_seed)
    length_test, length_train = len(labels_test), len(labels_train)
    good_test, good_train = sum(labels_test), sum(labels_train)
    
    # save as pickle files

    with open(savefile + "/texts_train.pickle", "wb") as sf:
        pickle.dump(texts_train, sf, protocol=pickle.HIGHEST_PROTOCOL)
    with open(savefile + "/texts_test.pickle", "wb") as sf:
        pickle.dump(texts_test, sf, protocol=pickle.HIGHEST_PROTOCOL)
    with open(savefile + "/labels_train.pickle", "wb") as sf:
        pickle.dump(labels_train, sf, protocol=pickle.HIGHEST_PROTOCOL)
    with open(savefile + "/labels_test.pickle", "wb") as sf:
        pickle.dump(labels_test, sf, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    # save dataset info
    with open(savefile + "/dataset_info.txt", "w" ) as info:
        info.write(savefile + "\nSizefactor: {} \nMaximal Sampling Size (NoGood Data): {} \nloc_factor: {} \ntest_data_size (factor): {} \n\
Testdata (good/all) {}/{} \nTraindata (good/all) {}/{} \nRandom_seed: {} \nSample type: {}".format(sizefactor, max_sampling_size, 
                   loc_factor, test_size, good_test, length_test, good_train, length_train, random_seed, sample_type))

    
def load_dataset(savefile):
    # load traintestsplit dataset
    with open(savefile + "/texts_train.pickle", "rb") as sf:
        texts_train = pickle.load(sf)
    with open(savefile + "/texts_test.pickle", "rb") as sf:
        texts_test = pickle.load(sf)
    with open(savefile + "/labels_train.pickle", "rb") as sf:
        labels_train = pickle.load(sf)
    with open(savefile + "/labels_test.pickle", "rb") as sf:
        labels_test = pickle.load(sf)
    return texts_train, texts_test, labels_train, labels_test



def create_combined_dataset(meta, sizefactor, savefile, max_sampling_size = 150000, 
                            loc_factor = 1.1, random_seed = random_seed, sample_type = "basic"):
    # same as create_dataset, but without traintestsplit, creates a combined dataset that is only split later
    os.mkdir(savefile)
    if sample_type == "beta":
        g, ng, plot = sample_metadata(meta = meta, sizefactor = sizefactor, max_sampling_size = max_sampling_size, sampling_bin_size = 100, 
                                loc_factor = loc_factor, no_shorter = True, plot = True)
    if sample_type == "basic":
        g, ng, plot = sample_meta_basic(meta = meta, sizefactor = sizefactor, max_sampling_size = max_sampling_size, sampling_bin_size = 100, 
                                no_shorter = True, plot = True)
    else: 
        print("wrong sample type")
        return 1
    plot.savefig(savefile + "/distribution_plot.png")
    g_data = load_articles(g)
    ng_data = load_articles(ng)
    labels = g_data[0] + ng_data[0]
    texts = g_data[1] + ng_data[1]
    labels, texts = shuffle(labels, texts, random_state=random_seed)
    length_all = len(labels)
    length_good = len(g_data[0])
    with open(savefile + "/texts.pickle", "wb") as sf:
        pickle.dump(texts, sf, protocol=pickle.HIGHEST_PROTOCOL)
    with open(savefile + "/labels.pickle", "wb") as sf:
        pickle.dump(labels, sf, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(savefile + "/dataset_info.txt", "w" ) as info:
        info.write(savefile + "\nSizefactor: {} \nMaximal Sampling Size (NoGood Data): {} \nloc_factor: {} \n\
Datalength (good/all) {}/{} \nRandom_seed: {} \nSample type: {}".format(sizefactor, max_sampling_size, 
                   loc_factor, length_good, length_all, random_seed, sample_type))

        
def load_combined_dataset(savefile):
    # loads the combined dataset
    with open(savefile + "/texts.pickle", "rb") as sf:
        texts = pickle.load(sf)
    with open(savefile + "/labels.pickle", "rb") as sf:
        labels = pickle.load(sf)
    return texts, labels


##############################################################################
# Application: Uncommenting this creates the 4 sampled datasets described in the report

# meta = load_metadata()
# create_dataset(meta, sizefactor = 1, savefile = "wikidset_1_basic", max_sampling_size=100000, 
#                loc_factor = 1.1, test_size = 0.2, random_seed = random_seed, sample_type = "basic")
# create_dataset(meta, sizefactor = 2, savefile = "wikidset_2_basic", max_sampling_size=100000, 
#                loc_factor = 1.1, test_size = 0.2, random_seed = random_seed, sample_type = "basic")
# create_dataset(meta, sizefactor = 5, savefile = "wikidset_5_basic", max_sampling_size=100000, 
#                loc_factor = 1.1, test_size = 0.2, random_seed = random_seed, sample_type = "basic")
# create_dataset(meta, sizefactor = 10, savefile = "wikidset_10_basic", max_sampling_size=100000, 
#                loc_factor = 1.1, test_size = 0.2, random_seed = random_seed, sample_type = "basic")

# create_combined_dataset(meta, sizefactor = 1, savefile = "wikidset_comb_1_basic", max_sampling_size=100000, 
#                         loc_factor = 1.1, random_seed = random_seed, sample_type = "basic")
# create_combined_dataset(meta, sizefactor = 2, savefile = "wikidset_comb_2_basic", max_sampling_size=100000, 
#                         loc_factor = 1.1, random_seed = random_seed, sample_type = "basic")
# create_combined_dataset(meta, sizefactor = 5, savefile = "wikidset_comb_5_basic", max_sampling_size=100000, 
#                         loc_factor = 1.1, random_seed = random_seed, sample_type = "basic")
# create_combined_dataset(meta, sizefactor = 10, savefile = "wikidset_comb_10_basic", max_sampling_size=100000, 
#                         loc_factor = 1.1, random_seed = random_seed, sample_type = "basic")

# ttrain, ttest, ltrain, ltest = load_dataset(savefile = "wikidset_1")
# texts, labels = load_combined_dataset(savefile = "wikidset_comb_1")