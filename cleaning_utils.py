# -*- coding: utf-8 -*-
"""
Created on Fri Sep 03 15:52:34 2021

@author: rapha
"""

import re
import nltk
    

def text_tokenization(text):
    # Replace all punctuation by dots
    text = re.sub(r'\.(?=[^ \W\d])', '. ', str(text))
    
    # Replace multiple whitespaces by single whitespace (Strip)
    text  = re.sub('\s+',' ',text)
    
    # Repeat Lowercasing from Parsing
    text = text.lower()
    
    # Tokenize Text with split
    tokens = text.split(" ")
    
    return tokens


def remove_stopwords(tokens, stopwords):
    #remove stopwords from stopword list
    red_tokens = []
    for word in tokens:
        if word in stopwords:
            red_tokens.append(word)
    return red_tokens


def sl(tokens, stemming = True, lemmatizing = True):
    # stemming and lemmatizing using nltk
    stemm = nltk.stem.porter.PorterStemmer()
    lemma = nltk.stem.wordnet.WordNetLemmatizer()
    sl = []
    for tok in tokens:
        if stemming: 
            tok = stemm.stem(tok)
        if lemmatizing:
            tok = lemma.lemmatize(tok)
        sl.append(tok)
    return sl


def cleaning(text, sl, language = "english"):
    """
    

    Parameters
    ----------
    text : string
        Text to be cleaned.
    sl : Bool
        Select if text should be stemmed and lemmatized.

    Returns
    -------
    Text.

    """
    tokens= text_tokenization(text)
    stopwords = nltk.corpus.stopwords.words(language)

    tokens = remove_stopwords(tokens, stopwords)
    if sl == True:
        tokens = sl(tokens, stemming = True, lemmatizing = True)
    text = " ".join(tokens)

    return text
        
