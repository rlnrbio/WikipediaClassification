# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 16:08:40 2021

@author: rapha
"""
from tensorflow.keras import models, layers, preprocessing

from sklearn.model_selection import train_test_split
import pickle
import sklearn.metrics as metrics
from dataloading import load_combined_dataset
import gensim.downloader as gensim_api
import numpy as np

length = 50

# code partially adapted from https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
def create_sequence(corpus, top=None, oov=None, maxlen=None, fitted_tokenizer=None ):    
    #adapted from text2seq
    #train tokenizer
    if fitted_tokenizer is None:
        tokenizer = preprocessing.text.Tokenizer(num_words=top, lower=True, split=' ', char_level=False, oov_token=oov,
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(corpus)
    #use existing tokenizer for Test data
    else:
        tokenizer= fitted_tokenizer
    #create vocabulary
    dic_vocabulary = tokenizer.word_index
    #create tokens
    lst_tokens = tokenizer.texts_to_sequences(corpus)

    ## padding/truncating sequence
    X =  preprocessing.sequence.pad_sequences(lst_tokens, maxlen=maxlen, padding="post", truncating="post")

    return {"X":X, "tokenizer":tokenizer, "dic_vocabulary":dic_vocabulary} if fitted_tokenizer is None else X
        

savefile = "wikidset_comb_1_basic"
split = 0.2
random_seed = 42

# load combined dataset
texts, labels = load_combined_dataset(savefile = savefile)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=split, random_state=random_seed)

dictionary = create_sequence(corpus=X_train, top=None, oov="NaN", maxlen=length)

X_train, tokenizer, dic_vocabulary = dictionary["X"], dictionary["tokenizer"], dictionary["dic_vocabulary"]

# Preprocess Test with the same tokenizer
X_test = create_sequence(corpus=X_test, fitted_tokenizer=tokenizer, maxlen=X_train.shape[1])

tokens_dict = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, 
               "tokenizer": tokenizer, "dic_vocabulary": dic_vocabulary}


# save tokenized and truncated articles to pickle
with open(savefile + "/tokenized_t2s_" + str(length) + ".pickle", 'wb') as handle:
    pickle.dump(tokens_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

##############################################################################

def load_tokens(savefile):
    with open(savefile + "/tokenized_t2s_" + str(length) + ".pickle", "rb") as sf:
        tokens = pickle.load(sf)
    X_train = tokens["X_train"]
    X_test = tokens["X_test"]
    y_train = tokens["y_train"]
    y_test = tokens["y_test"]
    dic_vocabulary = tokens["dic_vocabulary"]
    return X_train, X_test, y_train, y_test, dic_vocabulary

X_train, X_test, y_train, y_test, dic_vocabulary = load_tokens(savefile)

# create vocabulary embeddings using glove-wiki-gigaword-50
nlp = gensim_api.load("glove-wiki-gigaword-50")
embeddings = np.zeros((len(dic_vocabulary)+1, nlp.vector_size))
for word,idx in dic_vocabulary.items():
    # Only add embedding if it exists in the model
    try:
        embeddings[idx] =  nlp[word]
    except:
        pass


# Embedding network with Bi-LSTM 
x_in = layers.Input(shape=(X_train.shape[1],))
# embedding
x = layers.Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings],
                     input_length=X_train.shape[1], trainable=False)(x_in)
## 2 layers of bidirectional lstm
x = layers.Bidirectional(layers.LSTM(units=X_train.shape[1], dropout=0.2, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(units=X_train.shape[1], dropout=0.2))(x)
## final dense layers
x = layers.Dense(64, activation='relu')(x)
y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)

model = models.Model(x_in, y_out)
model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.summary()
y_test = np.array(y_test, dtype = "int") 
y_train = np.array(y_train, dtype = "int")
training = model.fit(X_train, y_train, batch_size = 256, epochs = 10,)


# evaluation 
predicted_probs = model.predict(X_test)
predicted = np.argmax(predicted_probs, axis = 1)

score = metrics.accuracy_score(y_test, predicted)
report = metrics.classification_report(y_test, predicted)
with open(savefile + "/" + "ml_model_" + str(length) + ".txt", "w" ) as info:
    info.write(report)
