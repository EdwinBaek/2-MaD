from __future__ import print_function
import time
from datetime import timedelta
import os

print("CUDA_VISIBLE_DEVICES 0 setting")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import FastText
from gensim.models import word2vec
import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding
from tensorflow.keras.models import load_model
import csv

def all_data_name_maker(label_file):
    # make all_data.txt file
    all_data_element = open('./data/train/BSY_alldata_name.txt', 'a', encoding='utf-8')
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)
    for line in csv_reader:
        try:
            # open opcode data
            op_file = open('./dataset/BSY_opcode/%s.txt' % line[0], 'r', encoding='utf-8')
            all_data_element.write(str(line[0]) + '##' + line[1] + '\n')
            op_file.close()

        except:
            pass
    csv_data.close()


def testset_namelist_maker():
    mal_cnt = 0
    ben_cnt = 0
    testset_name_list = []
    with open('./data/train/BSY_alldata_name.txt', 'r', encoding='utf-8') as all_data:
        for line in all_data:
            line = line.strip()
            name, label = line.split('##', 1)
            if label == '0':
                ben_cnt += 1
                if ben_cnt <= 11917:
                    pass
                elif ben_cnt > 11917 and ben_cnt <= 13407:
                    pass
                else:
                    testset_name_list.append(name)
            elif label == '1':
                mal_cnt += 1
                if mal_cnt <= 13099:
                    pass
                elif mal_cnt > 13099 and mal_cnt <= 14737:
                    pass
                else:
                    testset_name_list.append(name)

    return testset_name_list


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('##')
                if content:
                    contents.append(content.split(' '))
                    labels.append(label)
            except:
                pass
    return contents, labels


def read_vocab(vocab_dir):
    """word to id list generation"""
    with open_file(vocab_dir) as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id


def read_category():
    """class to id"""
    categories = ['0', '1']

    categories = [x for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """file to id"""
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        tempo = []
        for x in contents[i]:
            if x in word_to_id:
                tempo.append(word_to_id[x])

        data_id.append(tempo)
        label_id.append(cat_to_id[labels[i]])

    x_len = []
    for i in data_id:  # i is all contents of one file
        if len(i) < 600:
            x_len.append(len(i))
        else:
            x_len.append(600)

    # pad_sequences Unified length
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, maxlen=max_length, padding='post', truncating='post')
    # y_onehot = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    label_data = np.array(label_id)
    return x_pad, label_data, x_len


# embedding model load
def embedding_load(model_name, vocab_dir, model_select, embedding_dim):
    print("embedding model loading...")

    # changable
    word_vocab_name = '%s' % vocab_dir
    word2vec_file_name = './embedding_model/%s' % model_name

    with open(word_vocab_name) as fp:
        words = [_.strip() for _ in fp.readlines()]

    # changable
    embedding_matrix = np.zeros((len(words), embedding_dim))  # 300 600 900
    if model_select == 'fasttext':
        print("fasttext model loading...")
        print("vocab name = %s" % str(vocab_dir))
        print("word2vec_file_name = %s" % str(word2vec_file_name))
        model = FastText.load(word2vec_file_name)
    else:
        print("word2vec model loading...")
        print("vocab name = %s" % str(vocab_dir))
        print("word2vec_file_name = %s" % str(word2vec_file_name))
        model = word2vec.Word2Vec.load(word2vec_file_name)

    for i in range(len(words)):
        if words[i] in model.wv.vocab:
            embedding_vector = model.wv[words[i]]
            embedding_matrix[i] = embedding_vector

    print("embedding model load Done!")

    return model, embedding_matrix


def test_model(seq_length, test_dir, vocab_dir, test_batch, model_select, model_name, embedding_dim, method):
    TESTSET_NAME_LIST = testset_namelist_maker()
    print(TESTSET_NAME_LIST)
    print(len(TESTSET_NAME_LIST))

    TN, FN, TP, FP = 0, 0, 0, 0
    TN_index_list, FN_index_list, TP_index_list, FP_index_list = [], [], [], []

    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    vocab_size = len(words)

    x_test, y_test, x_test_length = process_file(test_dir, word_to_id, cat_to_id, seq_length)
    print("y_true 값")

    # load embedding model
    EBD_model, embedding_matrix = embedding_load(model_name, vocab_dir, model_select, embedding_dim)

    print('embedding_matrix shape : ' + str(embedding_matrix.shape))
    print('x_test shape : ' + str(x_test.shape))
    print('y_test shape : ' + str(y_test.shape))

    LSTM_model = load_model('./model/BSY_fasttext_model_100_2_layer_256_cell1_128_cell2_15-0.1545.h5')
    print("test start!")
    with tf.device('/gpu:0'):
        y_pred = LSTM_model.predict(x_test)
    print("Test Done")
    y_pred_bucket = []
    benign_pred_name = []
    y_true_list = y_test.tolist()
    y_pred_list = y_pred.tolist()

    for i in range(len(y_true_list)):
        if y_pred_list[i][0] >= 0.5:
            y_pred_bucket.append(1)
        else:
            y_pred_bucket.append(0)
            benign_pred_name.append(TESTSET_NAME_LIST[i])

    for i in range(len(y_true_list)):
        if y_pred_bucket[i] == y_true_list[i]:    # correct answer
            if y_true_list[i] == 1:               # if true is malware
                TP += 1
                TP_index_list.append(i)
            elif y_true_list[i] == 0:             # if true is benign
                TN += 1
                TN_index_list.append(i)
        elif y_pred_bucket[i] != y_true_list[i]:    # Fail to answer
            if y_true_list[i] == 1:               # if true is malware, predict is benign because fail to answer
                FN += 1
                FN_index_list.append(i)
            elif y_true_list[i] == 0:             # if true is benign, pred is malware
                FP += 1
                FP_index_list.append(i)
    '''
    print("TP")
    print(TP)
    print("TN")
    print(TN)
    print("FN")
    print(FN)
    print("FP")
    print(FP)


    print("TP_index_list")
    print(TP_index_list)
    print("TN_index_list")
    print(TN_index_list)
    print("FN_index_list")
    print(FN_index_list)
    print("FP_index_list")
    print(FP_index_list)

    print("benign_pred_name")
    print(benign_pred_name)
    print("benign_pred_name length")
    print(len(benign_pred_name))
    '''
    benign_predicted_txt = open('./data/train/predict_to_benign.txt', 'w', newline='', encoding='utf-8')
    for i in range(len(benign_pred_name)):
        benign_predicted_txt.write(benign_pred_name[i] + '\n')

    print("clear")



if __name__ == '__main__':
    '''
    train_label = './data/label/trainSet.csv'
    pre_label = './data/label/preSet.csv'
    final1_label = './data/label/finalSet1.csv'
    final2_label = './data/label/finalSet2.csv'

    all_data_name_maker(train_label)
    all_data_name_maker(pre_label)
    all_data_name_maker(final1_label)
    all_data_name_maker(final2_label)
    '''
    # hyperparam
    seq_length = 600
    test_batch = 8

    base_dir = './data/train'
    test_dir = os.path.join(base_dir, 'BSY_test.txt')
    vocab_dir = os.path.join(base_dir, 'BSY_word_vocab.txt')

    test_model(seq_length, test_dir, vocab_dir, test_batch,
               model_select='fasttext', model_name='BSY_fasttext_model_100', embedding_dim=100, method='BSY')

