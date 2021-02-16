#!/usr/bin/env python
# coding: utf-8

# Stopword preprocessing
import re
import numpy as np
import collections
from collections import Counter  # Class Counting stars
import os
import nltk
from keras.preprocessing import sequence
import sklearn.preprocessing
from keras.layers import Embedding, SpatialDropout1D, Conv1D, GlobalMaxPool1D, MaxPool1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Flatten, Dropout, LeakyReLU
import keras
import tensorflow as tf
from keras import backend as K
import keras.metrics
from sklearn import metrics


def stop_word(product_names, stopword, new):

    for name in product_names:
        for i in range(0,len(stopword)):
            k = re.sub(pattern=stopword[i] ,repl=' ', string=name)
            if k != name:
                name=k
        new.append(name)
    return new


# Plural word singularization

def words_to_word(product_names, mapping_dict):
    keys_list = list(mapping_dict.keys())
    replace_list = list(mapping_dict.values())

    tmp= list()
    for name in product_names:
        for i in range(0,len(mapping_dict)):
            k = re.sub(pattern=keys_list[i] ,repl=replace_list[i], string=name)
            if k != name:
                name=k
        tmp.append(name)
    return tmp


# Data Cleansing
# 1) Same product name, different HSCODE --> Replaced with the largest number of Count
def data_cleansing(hscode, product_names):
    # Create product name vocabulary
    product_names_dict = dict()
    limit_range = len(hscode)
    for i in range(0, limit_range):
        data_names = product_names[i]
        name_index = product_names_dict.get(data_names)
        if name_index is None:
            name_index = list()
            name_index.append(i)
            product_names_dict[data_names] = name_index
        else:
            name_index.append(i)

# If there is a different hscode with the same product name, it is replaced with the largest count
    for name in list(set(product_names[0:limit_range])):
        hscode_index = list()
        hscode_index.extend(product_names_dict.get(name))
        diff_hscode = list()
        if len(hscode_index) != 1:
            for i in hscode_index:
                diff_hscode.append(hscode[i])
            hscode_counter = Counter(diff_hscode)
            if len(hscode_counter) > 1:
                for i in hscode_index:
                    hscode[i] = max(hscode_counter.keys(), key=(lambda k: hscode_counter[k]))

    hscode_result = list()
    product_names_result = list()

    for i in range(0, len(hscode)):
        hscode_result.append(hscode[i])
        product_names_result.append(product_names[i])

    return hscode_result, product_names_result


# 2) Remove data with HSCODE that does not exist and blank data for product name

def delete_strange(hscode, product_names):

# Remove the code weird
    strange = ['XX', '00', '99', '98']

    strange_remove = hscode
    strange_ind = []
    for i in range(len(strange_remove)):
        if strange_remove[i][0:2] in strange:
            strange_ind.append(i)
        if strange_remove[i] == '0':
            strange_ind.append(i)

    hscode_strange = np.delete(hscode, strange_ind)
    product_names_strange = np.delete(product_names, strange_ind)

    # Remove data with spaces in the product name
    blank_remove = product_names
    blank_ind = []
    for i in range(len(blank_remove)):
        if blank_remove[i] == '':
            blank_ind.append(i)

    hscode_result = np.delete(hscode_strange, blank_ind)
    product_names_result = np.delete(product_names_strange, blank_ind)

    return hscode_result, product_names_result


def sequence_processing(product_names, hscode, num_recs, word2index):
    X = np.empty((num_recs,), dtype=list)  # (99992,~) 의 벡터 만들기
    y = np.zeros((num_recs,))
    i = 0

    for sentence, label in zip(product_names, hscode):
        words = nltk.word_tokenize(sentence)  # Tokenization one sentence at a time
        seqs = []
        for word in words:  # ZINC, WASTE, SCRAP Selected in order
            if word in word2index:  # If the word is in word2index, add the index of that word to seqs
                seqs.append(word2index[word])  # word2index --> sentence is indexed
            else:
                seqs.append(word2index["UNK"])  # UNK without that word
        # print(seqs)
        X[i] = seqs  # X[0] = [ZINC, WASTE, SCRAP] Index number corresponding to [10,  3000,   9503]
        y[i] = int(label)  # hs code
        i += 1

    return X, y


# One-hot encoding

def one_hot(y):
    y = y.astype(dtype='int64')
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(y) + 1))
    one_hot_y = label_binarizer.transform(y)
    return one_hot_y


# Oversampling
def oversampling(hscode_unit, hscode_cleansing, product_names_cleansing):
    hscode_count = Counter(hscode_unit)

    hscode_cleansing = list(hscode_cleansing)
    product_names_cleansing = list(product_names_cleansing)

    under_count = []
    copy_num = []
    hscode_kind = list(set(list(hscode_unit)))

    for i in hscode_kind:
        if hscode_count[i] < 700:
            copy_num.append(int(700 / (hscode_count[i] + 1)))
            under_count.append(i)

    ## Copying process
    for i in range(len(under_count)):
        index = []
        for a in range(len(hscode_cleansing)):
            if int(hscode_cleansing[a][0:2]) == under_count[i]:
                index.append(a)
        print(len(index))

        copy_hscode = []
        copy_pro_name = []
        for ind in index:
            copy_hscode.append(hscode_cleansing[ind])
            copy_pro_name.append(product_names_cleansing[ind])

        if copy_num[i] != 1:
            copy_hscode = copy_hscode * (copy_num[i] - 1)
            copy_pro_name = copy_pro_name * (copy_num[i] - 1)

        hscode_cleansing.extend(copy_hscode)
        product_names_cleansing.extend(copy_pro_name)

        # For confirmation
    hscode_unit_over = []
    for row in range(0, len(hscode_cleansing)):
        hscode_unit_over.append(hscode_cleansing[row][0:6])  # In case of oversampling, change to hscode_unit_over to proceed with code

    return hscode_unit_over





# Recall
def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max Other edges are cut off
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # Set the actual value to 0 (Negative) or 1 (Positive)
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # Set the predicted value to 0 (Negative) or 1 (Positive)

    # True positive is when both the actual and predicted values are 1 (positive).
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Negative) = Actual value is 1 (Positive) total
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' Add small numbers as a precaution
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall

# Precision
def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max Other edges are cut off
    # round : Round
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # Set the predicted value to 0 (Negative) or 1 (Positive)
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # Set the actual value to 0 (Negative) or 1 (Positive)

    # True Positive Is when both the actual and predicted values are 1 (Positive).
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon() Adds a small number to prevent'divide by zero error'
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision


# F1-score
def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon() adds a small number to prevent'divide by zero error'
    _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())

    # return a single tensor value
    return _f1score


### Top 1,3,5 Acc, Recall, Precision, F1-score
def topN_score(predict_proba, y_test):
    ### 클래스 재부여
    y_class = []
    for test in y_test:
        cnt = 0
        for i in test:
            if i == 1:
                y_class.append(cnt)
            cnt += 1

    # # TOP  - 1 score
    # cnt = 0
    # for i in range(len(y_class)):
    #     top_1_label = []  # Initialize list
    #     top_1_label = predict_proba[i].argsort()[::-1][0]  # Put it in the top 3 list
    #
    #     if y_class[i] in top_1_label:
    #         cnt += 1
    # print('TOP 1 Accuracy:', cnt / len(y_class))

    # TOP  - 3 score
    cnt = 0
    for i in range(len(y_class)):
        top_3_label = []  # Initialize list
        top_3_label = predict_proba[i].argsort()[::-1][0:3]  # Put it in the top 3 list

        if y_class[i] in top_3_label:
            cnt += 1
    print('TOP 3 Accuracy:', cnt / len(y_class))

    # TOP  - 5 score
    cnt = 0
    for i in range(len(y_class)):
        top_5_label = []  # # Initialize list
        top_5_label = predict_proba[i].argsort()[::-1][0:5]  # Put it into the top 5 list

        if y_class[i] in top_5_label:
            cnt += 1
    print('TOP 5 Accuracy:', cnt / len(y_class))

    y_pred_first = []
    for i in range(len(y_class)):
        y_pred_first.append(predict_proba[i].argsort()[::-1][0])

    y_pred_second = []
    for i in range(len(y_class)):
        y_pred_second.append(predict_proba[i].argsort()[::-1][1])

    y_pred_third = []
    for i in range(len(y_class)):
        y_pred_third.append(predict_proba[i].argsort()[::-1][2])

    y_pred_fourth = []
    for i in range(len(y_class)):
        y_pred_fourth.append(predict_proba[i].argsort()[::-1][3])

    y_pred_fifth = []
    for i in range(len(y_class)):
        y_pred_fifth.append(predict_proba[i].argsort()[::-1][4])

    report_first = metrics.classification_report(y_class, y_pred_first, digits=3, output_dict=True)
    report_second = metrics.classification_report(y_class, y_pred_second, digits=3, output_dict=True)
    report_third = metrics.classification_report(y_class, y_pred_third, digits=3, output_dict=True)
    report_fourth = metrics.classification_report(y_class, y_pred_fourth, digits=3, output_dict=True)
    report_fifth = metrics.classification_report(y_class, y_pred_fifth, digits=3, output_dict=True)

    report_first_avg = report_first['weighted avg']
    report_second_avg = report_second['weighted avg']
    report_third_avg = report_third['weighted avg']
    report_fourth_avg = report_fourth['weighted avg']
    report_fifth_avg = report_fifth['weighted avg']

    # Top1, 3 F1 score

    print("Top1 F1 : ", report_first_avg['f1-score'])

    print("Top3 F1 : ", report_first_avg['f1-score'] + report_second_avg['f1-score'] + report_third_avg['f1-score'])

    # Top5 F1 score

    print("Top5 F1 : ", report_first_avg['f1-score'] + report_second_avg['f1-score'] +
          report_third_avg['f1-score'] + report_fourth_avg['f1-score'] + report_fifth_avg['f1-score'])

    # Top1, 3 recall
    print("Top1 recall : ", report_first_avg['recall'])

    print("Top3 recall : ", report_first_avg['recall'] + report_second_avg['recall'] + report_third_avg['recall'])

    # Top5 recall

    print("Top5 recall : ", report_first_avg['recall'] + report_second_avg['recall'] +
          report_third_avg['recall'] + report_fourth_avg['recall'] + report_fifth_avg['recall'])

    # Top1, 3 precision
    print("Top1 precision : ", report_first_avg['precision'])

    print(
    "Top3 precision : ", report_first_avg['precision'] + report_second_avg['precision'] + report_third_avg['precision'])

    # Top5 precision

    print("Top5 precision : ", report_first_avg['precision'] + report_second_avg['precision'] +
          report_third_avg['precision'] + report_fourth_avg['precision'] + report_fifth_avg['precision'])
