# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 18:11:57 2021

@author: rapha
"""
# -*- coding: utf-8 -*
# This takes very long to run (several days)
import mwxml
#package from wikimedia foundation
import glob
import bz2
import pandas as pd
import re
import numpy as np
from datetime import datetime
import pickle
import sys

#sourcefolder for Wikipedia dump
src = "wikisource"
srcfile = "enwiki-20210901-pages-articles-multistream*.xml*.bz2"
#Folder for storing results
res = "wikiresults"

def check_redirect(text):
    #check if article is redirect
    if text != None:
        red = "#REDIRECT" in text
    else: red = True
    return red

def basic_text_processing(text):
    #check if article is good article
    good = "{{good article}}" in text
    
    #remove newlines:
    text = re.sub("\n", " ", text)
    
    #remove control characters:
    cc = r"[^\w\s\{\}\/\:]"
    text  = re.sub(cc,' ',text)
    
    # remove multiple whitespaces
    text  = re.sub('\s+',' ',text)
    
    #remove all notifications
    text = re.sub('\{\{.*?\}\}', '', text)
    
    textlength = len(text)
    
    return good, textlength, text

paths = glob.glob(src + '/' + srcfile)

#create metadata DF
metadata = pd.DataFrame(columns = ["ProjectID", "InFile", "WikiID", "title", "textlength", "goodArticle", "lastChange", "random"])

#write protocoll
with open("wikiresults/protocol.csv", "a") as file_object:
    file_object.write("Path, AllArticles, CountedArticles, GoodArticles, TotalLetters, Filename, ProcessingDur \n")
    
# count values
projectID_count = 0
fileID = 0
count_articles = 0
curr_filesize = 0
count_goodarticles = 0
total_letters = 0
texts = dict()

#for combined File (18 GB)
dump = mwxml.Dump.from_file(bz2.open(paths[0]))
start = datetime.now()
for page in dump:
    #iterate through Wikipedia XML files
    for revision in page:
        count_articles += 1
        text = revision.text
        # only use non-redirect articles
        if check_redirect(text) == False:
            projectID_count += 1
            wikiID = page.id
            title = page.title
            lastchange = revision.timestamp
            good, textlength, text = basic_text_processing(revision.text)
            count_goodarticles += int(good)
            total_letters += textlength
            #give random number to articles for sub-datasets
            random = np.random.randint(100)
            # create metadata to article
            metadata = metadata.append({"ProjectID": projectID_count, "InFile": fileID, "WikiID": wikiID, "title": title, 
                         "textlength": textlength, "goodArticle": good, "lastChange": lastchange, "random": random }, ignore_index = True)
            # save text
            texts[projectID_count] = text
            # save file after given size, report progress incl. processing time
            if projectID_count%1000 == 0:
                print(title, wikiID)
                print(str(projectID_count) +  ", 1000_check, filesize = " + str(sys.getsizeof(texts)))
                if sys.getsizeof(texts)>= 5000000:
                    filename = 'wikiresults/wiki' + str(fileID).zfill(2) + '.pickle'
                    print(filename + " written")
                    with open(filename, 'wb') as handle:
                        pickle.dump(texts, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    fileID += 1
                    texts = dict()
                    end = datetime.now()
                    dur = end-start
                    print(dur)
                    start = datetime.now()
            
# save protocoll and metadata
with open("wikiresults/protocol.csv", "a") as file_object:
    file_object.write(paths[0] + "," + str(count_articles) + "," + str(projectID_count) + "," + str(count_goodarticles) 
                      + "," + str(total_letters)  + "," + filename + "," + str(dur) +'\n' )
           
metadata.to_csv("wikiresults/metadata.csv")
    

###########################
# # Load data (deserialize)
# with open('wwiki01.pickle', 'rb') as handle:
#     texts = pickle.load(handle)
        
            
