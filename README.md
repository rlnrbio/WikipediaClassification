# WikipediaClassification
This repository contains the code a trial Wikipedia article quality classification project, 09/2021


parsing.py: Parse data from Wikipedia data dump, data available from https://dumps.wikimedia.org/enwiki/20210901/

dataloading.py: Script to create datasets and contain functions for simple dataloading

cleaning_utils.py: Some functions to clean data used during dataloading

statistics.py: Script to analyze data and data distribution, optional, not required for analysis

textvectorization.py: Script to create tf-idf text vectorizations of parsed and cleaned Wikipedia articles

classifComp.py: Script to compare the performance of some of the most common classical ML classifiers on tf-idf data

logisticRegression.py: More detailed analysis including feature importance of logistic Regression classification

randomForest.py: More detailed analysis including feature importance of random Forest classification

gloveModel.py: Script to run a more complex Glove (W2V derivate) and DNN classification model


You also find the final report presenting the initial results of the project.
