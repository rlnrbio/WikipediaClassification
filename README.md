# WikipediaClassification
This repository contains the code for a trial Wikipedia article quality classification project, 09/2021

## Idea
The idea behind this project was to build a Proof of Concept and to analyze how well conventional text classification algorithms as well as Neural Networks are able to evaluate the quality of Wikipedia articles automatically. 
For this, it utilizes articles as training and evaluation data that have manually been curated and have been assigned the "good article" batch by Wikipedia Editors. 
It is an example for the implementation of a simple, stand alone pipeline from data creation, curation and cleaning as well as analysis. The [report](https://github.com/rlnrbio/WikipediaClassification/blob/main/NLP_Report_Leuner_final.pdf) included in this repository contains a detailed description of data sources, data processing and analysis and can be used for further improvements of conventional text classification models.

## Code
- [parsing.py](https://github.com/rlnrbio/WikipediaClassification/blob/main/parsing.py): Parse data from Wikipedia data dump, data available from https://dumps.wikimedia.org/enwiki/20210901/
- [dataloading.py](https://github.com/rlnrbio/WikipediaClassification/blob/main/dataloading.py): Script to create datasets and contain functions for simple dataloading
- [cleaning_utils.py]((https://github.com/rlnrbio/WikipediaClassification/blob/main/cleaning_utils.py): Some functions to clean data used during dataloading
- [statistics.py](https://github.com/rlnrbio/WikipediaClassification/blob/main/statistics.py): Script to analyze data and data distribution, optional, not required for analysis
- [textvectorization.py](https://github.com/rlnrbio/WikipediaClassification/blob/main/textvectorization.py): Script to create tf-idf text vectorizations of parsed and cleaned Wikipedia articles
- [classifComp.py](https://github.com/rlnrbio/WikipediaClassification/blob/main/classifComp.py): Script to compare the performance of some of the most common classical ML classifiers on tf-idf data
- [logisticRegression.py](https://github.com/rlnrbio/WikipediaClassification/blob/main/logisticRegression.py): More detailed analysis including feature importance of logistic Regression classification
- [randomForest.py](https://github.com/rlnrbio/WikipediaClassification/blob/main/randomForrest.py): More detailed analysis including feature importance of random Forest classification
- [gloveModel.py](https://github.com/rlnrbio/WikipediaClassification/blob/main/gloveModel.py): Script to run a more complex Glove (W2V derivate) and DNN classification model


