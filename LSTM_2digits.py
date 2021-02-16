
# package import
from keras.layers import Embedding, SpatialDropout1D, Conv1D, GlobalMaxPool1D, MaxPool1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Flatten, Dropout, LeakyReLU, BatchNormalization, ReLU
from keras import regularizers

import keras
import keras.metrics
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import time
import copy
import nltk    # Tokenizer
import warnings   # Remove warning message
warnings.filterwarnings('ignore')

import collections   # Class Counting stars
import re   # Regular expression
from gensim.models import word2vec  # word2vec
from keras.preprocessing import sequence
import sklearn.preprocessing   # sklearn Data processing
from sklearn.model_selection import train_test_split
from keras import backend as K

config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=config)
K.set_session(sess)


# Load defined module
from modules import stop_word, words_to_word, data_cleansing, delete_strange, sequence_processing, one_hot, topN_score


# Data Load
limit_range = 99999
product_names_list = list()
with open('./product_name.txt', mode='r', encoding='utf-8-sig') as f:
    lines = f.readlines()
    for row in lines:
        product_names_list.append(row.strip())
product_names = np.array(product_names_list)

hscode_list = list()
with open('./hscode.txt', mode='r', encoding='utf-8-sig') as f:
    lines = f.readlines()
    for row in lines:
        hscode_list.append(row.strip())
hscode = np.array(hscode_list)



#StopWord Pretreatment
stopword = [' AND ', ' OF ',' IN ',' BRAND ', ' WITH ',' NO ',' HS ', ' CODE ',
        ' TO ',' CODE ',' CO ', ' PO ', ' AS ', ' LTD ', ' AT ', ' ONLY ',
        ' BY ', ' IS ', ' FROM ', ' TH E', ' THAT ',' BUT ', ' AT ',' PER ',
        ' FOR ',' TEL ',' INC ',' ON ',' OR ',
        ' WILL ',' BE ',' TOTAL ',' EXP ',' OUT ',' ALL ',' REF ']
new= []

product_names = stop_word(product_names, stopword, new)


# Multi-word singularization processing  Ex) Apples -> Apple
# Words to replace
mapping_dict= {"ZINCS": "ZINC","MECEDEZ": "MERCEDES","MERCEDEZ": "MERCEDES", "SPUNLACED": "SPUNLACE", "TYRE" : "TIRE",
        "MEN" : "MAN", "WOMEN" : "WOMAN", "PALLETS" : "PALLET", "CARTONS" : "CARTON","BAGS" : "BAG","PACKAGES" : "PACKAGE",
        "DRUMS" : "DRUM","PIECES" : "PIECE","CONTAINERS" : "CONTAINER","UNITS" : "UNIT","CASES" : "CASE", "DETAILS" : "DETAIL",
        "BOXES" : "BOX","FABRICS" : "FABRIC","TIRES" : "TIRE","PRODUCTS" : "PRODUCT","MANS" : "MAN","WOMANS" : "WOMAN",
        'ACCESSORIES':'ACCESSORY','CONTAINS' : 'CONTAIN','SHIPPERS' : 'SHIPPER','CRATES' : 'CRATE', 'GLOVES' : 'GLOVE',
        'LADIES' : 'LADY','BATTERIES' : 'BATTERY','WINES':'WINE','CANS' : 'CAN','TOYS' : 'TOY',"CONES" : "CONE", "BANANAS" : "BANANA",
        "COILS" : "COIL","TOWELS" : "TOWEL","REAMS" : "REAM","SHOES" : "SHOE","TANKS" : "TANK","GARMANTS" : "GARMENT",
        "DEGREES" : "DEGREE","CUSTOMS" : "CUSTOM","SHIPMENTS" : "SHIPMENT","CYLINDERS" : "CYLINDER","CHEMICALS" : "CHEMICAL",
        "CHIPS" : "CHIP","VEHICLES" : "VEHICLE","TUBES" : "TUBE","CAPS" : "CAP","BLOCKS" : "BLOCK","FLAKES" : "FLAKE",
        "MOTORCYCLES" : "MOTORCYCLE","BOTTLES" : "BOTTLE","PLANTS" : "PLANT","POUNDS" : "POUND","ITEMS" : "ITEM", "ENGINES" : "ENGINE",
        "SHOOTS" : "SHOOT","CONDITIONERS" : "CONDITIONER","WHEELS" : "WHEEL","STORES" : "STORE","GRAPES" : "GRAPE","LOGS" : "LOG",
        "ARTICLES" : "ARTICLE","CLOTHES" : "CLOTH","CLAYS" : "CLAY","RACKS" : "RACK","SAMPLES" : "SAMPLE","SUPPIES" : "SUPPLY", "COMPONENTS" : "COMPONENT",
        "MACHINES" : "MACHINE","TOOLS" : "TOOL","FILAMANTS" : "FILAMANT","BEANS" : "BEAN","BARS" : "BAR","WINGS" :"WING","COVERS" : "COVER",
        "GIRLS" : "GIRL","BOYS" : "BOY","PLATES" : "PLATE","NAILS" : "NAIL","TRUCKS" : "TRUCK","BUNDELS" : "BUNDEL","LEMONS" : "LEMON",
        "BOOKS" : "BOOK","RIBS" : "RIB","PACKETS" : "PACKET","BEVERAGES" : "BEVERAGE","LINES" : "LINE","FITTINGS" : "FITTING", "CARS" : "CAR",
        "FOODS" : "FOOD"}

product_names = words_to_word(product_names, mapping_dict)

# Data Cleansing

# 1) Same product name different HSCODE --> Count Replaced with the largest number

hscode_cleansing, product_names_cleansing = data_cleansing(hscode, product_names)

#2) Removing data with HSCODE that does not exist and blank data for product name

hscode_cleansing, product_names_cleansing = delete_strange(hscode_cleansing, product_names_cleansing)


#6 Extract large or medium classification among 6 digits

hscode_unit = list()
for row in range(0, len(hscode_cleansing)):
    hscode_unit.append(int(hscode_cleansing[row][0:2]))


# Sentence length, number of words without duplicates, number of sentences
nltk.download('punkt')
temp=list()   # To find out the length of the sentence

maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
for sentence in product_names_cleansing:
    words = nltk.word_tokenize(sentence)
    #temp.append(len(words)>15)
    if len(words) > maxlen:
        maxlen = len(words)  # the maximum number of words in a sentence
    for word in words:
        word_freqs[word] += 1  # frequency for each word
    num_recs += 1 # total number of records

print('Longest trade name length: {}, Word count without duplicates: {}, Product name Data number: {}'.format(maxlen, len(word_freqs), num_recs))


# Word2Vec Weight Matrix

# On the double list split
# Word2vecTo put it in the right shape

product_names_word2vec= list()
for i in product_names_cleansing:
    a= i.split(" ")
    product_names_word2vec.append(a)

embedding_size = 300    # Pre-composed using up to 40000 words
min_count = 1    # Use only words that appear at least n times
max_sentence_length= 38   # Maximum sentence length 50


# Word2Vec Embedding
w2v_model = word2vec.Word2Vec(product_names_word2vec, size=embedding_size, min_count= min_count)
print(w2v_model)

# Similar words
# w2v_model.wv.most_similar("BANANA")


# word2vec Vector value
w2v_weight = w2v_model.wv.vectors
w2v_weight.shape


# Word {Key:Value} Dictionary processing and processing when an existing word is entered
index2word = {i+2: w for i, w in enumerate(w2v_model.wv.index2word)}
index2word[0] = 'PAD'
index2word[1] = 'UNK'
word2index = {w: i for i, w in index2word.items() }
vocab_size = len(word_freqs) + 2


# Match the length of all sentences (Zero Padding)
X, y = sequence_processing(product_names_cleansing, hscode_unit, num_recs, word2index)
X = sequence.pad_sequences(X, maxlen=max_sentence_length, padding='post')

# X.shape, y.shape


# HSCODE One-hot encoding
one_hot_Y = one_hot(y)

# Train / Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, one_hot_Y, test_size=0.05, random_state=1)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# w2v_weight.shape, embedding_size, max_sentence_length

# Metric
from modules import recall, precision, f1score


# Building Network

# LSTM
model = Sequential()
model.add(
    Embedding(vocab_size, embedding_size, input_length=max_sentence_length, mask_zero=True, weights=[w2v_weight]))
model.add(LSTM(512, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(LSTM(256, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(98, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc", precision, recall, f1score])
model.summary()

BATCH_SIZE = 700
NUM_EPOCHS = 20


# Model Training
hist = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.1, shuffle=True)

# Model Evaluation
loss_test, acc_test, precision, recall, f1score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print('loss: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1score: {:.3f}'.format(loss_test, acc_test, precision, recall, f1score))

# Learning graph
fig, ax = plt.subplots(1,2, figsize=(18,6))

ax[0].plot(hist.history['loss'], 'b',label='Train loss')
ax[0].plot(hist.history['val_loss'], 'r', label='Val loss')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[0].legend(loc='upper left')
ax[0].set_xticks([0,5,10,15,20])

ax[1].plot(hist.history['acc'], 'b', label='Train acc')
ax[1].plot(hist.history['val_acc'], 'r', label='Val acc')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('accuracy')
ax[1].legend(loc='upper left')
ax[1].set_xticks([0,5,10,15,20])

plt.show()

# Top3, Top5 Score
predict_proba = model.model.predict(x_test)
topN_score(predict_proba, y_test) ;


# New input data input
new = np.empty((1, ), dtype= list)   # (1, ~) Create vector

new_input = 'BARVO NINE HUNDRED '
words = nltk.word_tokenize(new_input)
seq = []
for word in words:
  if word in word2index:
    seq.append(word2index[word])
  else:
    seq.append(word2index["UNK"])
new[0] = seq

new_input = sequence.pad_sequences(new, maxlen=max_sentence_length)

class_predict = model.model.predict(new_input)
class_index = class_predict[0].argsort()[::-1]


# percentage Top 5
for i in class_index[0:5]:
  print(i,  ' : ', class_predict[0][i])
