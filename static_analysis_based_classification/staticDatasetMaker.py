import csv
import os
#import statistics
from gensim.models import word2vec
from gensim.models import FastText
import logging
import numpy as np
import matplotlib.pyplot as plt

# make BSY_all_data.txt for train w2v & fasttext model
def all_dataMaker(label_file):
    # make all_data.txt file
    all_data_element = open('./dataset/BSY_all_data.txt', 'a', encoding='utf-8')
    csv_data = open(label_file, 'r', encoding='utf-8')
    opcode_vol_cnt = open('./dataset/BSY_all_data_opcode_cnt.txt', 'a', encoding='utf-8')
    csv_reader = csv.reader(csv_data)

    for line in csv_reader:
        try:
            # open opcode data
            op_file = open('./dataset/BSY_opcode/%s.txt' % line[0], 'r', encoding='utf-8')
            all_data_element.write(str(line[1]) + '##')

            i = 0
            # make contents of opcode part of all data
            for opline in op_file:
                # cutting over opcode mean
                if opline.find("?") != -1:
                    modified = opline.replace("?", "")
                    all_data_element.write(modified.strip() + ' ')
                    i += 1
                elif opline.find(" ") != -1:
                    modified = opline.replace(" ", "")
                    all_data_element.write(modified.strip() + ' ')
                    i += 1
                else:
                    all_data_element.write(opline.strip() + ' ')
                    i += 1

            all_data_element.write('\n')
            opcode_vol_cnt.write('%s opcode count : %d' % (line[0], i) + '\n')
            op_file.close()

        except:
            pass
    opcode_vol_cnt.close()
    csv_data.close()

def JJU_dataMaker(label_file):
    # make all_data.txt file
    all_data_element = open('./dataset/JJU1_all_data.txt', 'a', encoding='utf-8')
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)

    for line in csv_reader:
        try:
            # open opcode data
            op_file = open('./dataset/JJU1_opcode/%s.txt' % line[0], 'r', encoding='utf-8')
            all_data_element.write(str(line[1]) + '##')

            # make contents of opcode part of all data
            for opline in op_file:
                # cutting over opcode mean
                if opline.find("?") != -1:
                    modified = opline.replace("?", "")
                    all_data_element.write(modified.strip() + ' ')
                elif opline.find(" ") != -1:
                    modified = opline.replace(" ", "")
                    all_data_element.write(modified.strip() + ' ')
                else:
                    all_data_element.write(opline.strip() + ' ')

            all_data_element.write('\n')
            op_file.close()

        except:
            print(str(line) + ' error!')
            pass
    csv_data.close()
    all_data_element.close()

def JJU2_dataMaker(label_file):
    # line[0] : filename, line[1] : 0 or 1, line[2] : Packed or Unpacked, line[3] : entropy
    # make all_data.txt file
    all_data_element = open('./dataset/JJU2_all_data.txt', 'a', encoding='utf-8')
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)

    for line in csv_reader:
        try:
            if line[2] == 'Unpacked':
                # open opcode data
                op_file = open('./dataset/JJU2_opcode/%s.txt' % line[0], 'r', encoding='utf-8')
                all_data_element.write(str(line[1]) + '##')

                # make contents of opcode part of all data
                for opline in op_file:
                    # cutting over opcode mean
                    if opline.find("?") != -1:
                        modified = opline.replace("?", "")
                        all_data_element.write(modified.strip() + ' ')
                    elif opline.find(" ") != -1:
                        modified = opline.replace(" ", "")
                        all_data_element.write(modified.strip() + ' ')
                    else:
                        all_data_element.write(opline.strip() + ' ')

                all_data_element.write('\n')
                op_file.close()

        except:
            print(str(line) + ' error!')
            pass
    csv_data.close()
    all_data_element.close()

def all_dataMaker2(label_file, benign_cnt, malware_cnt):
    # make all_data.txt file
    all_data_element = open('./dataset/4_all_data.txt', 'a')
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)

    for line in csv_reader:
        try:
            # 두개 다 14859개로 1:1 비율이 맞으면 끝내기!
            if benign_cnt == 14859 and malware_cnt == 14859:
                print("malware_cnt = %d     benign_cnt = %d " % (malware_cnt, benign_cnt))
                break

            # benign 갯수랑 malware 갯수 16081 되면 안돌고 pass
            if line[1] == '0' and benign_cnt == 14859:
                continue
            elif line[1] == '1' and malware_cnt == 14859:
                continue

            # open opcode data
            op_file = open('./dataset/4_opcode/%s.txt' % line[0], 'r')
            all_data_element.write(str(line[1]) + '##')

            i = 0
            # make contents of opcode part of all data
            for opline in op_file:
                if i >= 5000:
                    break
                # cutting over opcode mean
                if opline.find('?') != -1:
                    opline.replace('?', '')
                    all_data_element.write(opline.strip() + ' ')
                    i += 1
                else:
                    all_data_element.write(opline.strip() + ' ')
                    i += 1

            all_data_element.write('\n')

            op_file.close()

            if line[1] == '0':
                benign_cnt += 1
            elif line[1] == '1':
                malware_cnt += 1
        except:
            pass
    csv_data.close()
    return benign_cnt, malware_cnt

# remove label at all_data text file
def label_del():
    train_label = open('./dataset/JJU1_all_data.txt', 'r')
    save = open('./data/train/JJU1_train_delet_label.txt', 'w')
    for line in train_label:
        train_list = []
        label, contents = line.split('##', 1)
        train_list.append(contents)
        train_list = ''.join(train_list)
        print(train_list, file=save, end='')
    del train_list

def train_embedding():
    if not os.path.exists('./data/train/JJU1_train_delet_label.txt'):
        label_del()
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus('./data/train/JJU1_train_delet_label.txt')

    # start word2vec train
    print("word2vec 100 train start!")
    model = word2vec.Word2Vec(sentences, iter=10, min_count=1, size=100, workers=4, window=5, sg=1)
    model.save('./embedding_model/JJU1_word2vec_model_100')
    print("Done train_word2vec 100")

    print("word2vec 200 train start!")
    model = word2vec.Word2Vec(sentences, iter=10, min_count=1, size=200, workers=4, window=5, sg=1)
    model.save('./embedding_model/JJU1_word2vec_model_200')
    print("Done train_word2vec 200")

    print("word2vec 300 train start!")
    model = word2vec.Word2Vec(sentences, iter=10, min_count=1, size=300, workers=4, window=5, sg=1)
    model.save('./embedding_model/JJU1_word2vec_model_300')
    print("Done train_word2vec 300")

    # start fasttext train
    print("fasttext 100 train start!")
    model2 = FastText(sentences, iter=10, min_count=1, size=100, workers=4, min_n=2, max_n=6)
    model2.save('./embedding_model/JJU1_fasttext_model_100')
    print("Done train_fasttext 100")

    print("fasttext 200 train start!")
    model2 = FastText(sentences, iter=10, min_count=1, size=200, workers=4, min_n=2, max_n=6)
    model2.save('./embedding_model/JJU1_fasttext_model_200')
    print("Done train_fasttext 200")

    print("fasttext 300 train start!")
    model2 = FastText(sentences, iter=10, min_count=1, size=300, workers=4, min_n=2, max_n=6)
    model2.save('./embedding_model/JJU1_fasttext_model_300')
    print("Done train_fasttext 300")

# build vocab.txt file
def build_vocab(model, vocab_dir):
    words = []
    # for i in range(len(model.wv.vocab)):
    for i in range(len(model.wv.vocab)):
        words.append(model.wv.index2word[i])

    # Add <pad> to word_vocab
    words = ['<PAD>'] + list(words)

    open(vocab_dir, mode='w', encoding='utf-8', errors='ignore').write('\n'.join(words) + '\n')

def MS_dataset_Maker(label_file):
    # make all_data.txt file
    all_data_element = open('./dataset/BSY_all_data_seq600.txt', 'a')
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)
    for line in csv_reader:
        try:
            # open opcode data
            op_file = open('./dataset/MS_dataset_opcode/%s.txt' % line[0], 'r')
            all_data_element.write(str(line[1]) + '##')

            # make contents of opcode part of all data
            i = 0
            for opline_char in op_file:
                # cutting over opcode mean
                if i >= 550:
                    break
                opline = opline_char.replace(' ', '')
                if opline.find('?') != -1:
                    opline.replace('?', '')
                    all_data_element.write(opline.strip() + ' ')
                    i += 1
                else:
                    all_data_element.write(opline.strip() + ' ')
                    i += 1


            op_file.close()

        except:
            pass

    csv_data.close()

'''
# make words name & vector dictionary
def make_vectorDict(model, vocab_dir, embedding_size):
    vector_list = []
    name_list = []
    with open(vocab_dir, mode='r', encoding='utf-8', errors='ignore') as fp:
        words = [_.strip() for _ in fp.readlines()]

    for i in range(len(words)):
        if words[i] in model.wv.vocab:
            name_list.append(words[i])
            embedding_vector = model.wv[words[i]]
            vector_list.append(sum(embedding_vector) / embedding_size)

    data_dict = dict(zip(name_list, vector_list))

    return data_dict

def vector_maker(model, vocab_data, all_vector_txt, label_file, embedding_size):
    data_dict = make_vectorDict(model, vocab_data, embedding_size)

    # make all_data_vector.txt file
    all_data_element = open(all_vector_txt, 'a')
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)
    for line in csv_reader:
        try:
            # open opcode data
            file = open('./dataset/thinkingmind/pre_opcode/%s.txt' % line[0], 'r')
            all_data_element.write(str(line[1]) + '##')

            # make contents of opcode part of all data
            i = 0
            for opline_char in file:
                # cutting over opcode mean
                if i >= 900:
                    break

                opline = opline_char.replace(' ', '')

                if opline.find('?') != -1:
                    opline.replace('?', '')
                    opcode_input = data_dict[opline.strip()]
                    all_data_element.write('%f ' % opcode_input)
                    i += 1
                else:
                    opcode_input = data_dict[opline.strip()]
                    all_data_element.write('%f ' % opcode_input)
                    i += 1

            file.close()
        except:
            pass

        try:
            # write API contents at all_data.txt
            file = open('./dataset/thinkingmind/pre_api/%s.txt' % line[0], 'r')
            k = 0
            # api 들을 list에 저장하여 all_data.txt에 넣을 contents 만들기
            for APIline in file:
                # cutting over API mean
                if k >= 100:
                    break
                API_input = data_dict[APIline.strip()]
                all_data_element.write('%f ' % API_input)
                k += 1

            all_data_element.write('\n')
            file.close()
        except:
            pass

    csv_data.close()

def make_vector_dataset(all_vector_txt, embedding_file_name):
    tempo = []
    with open(all_vector_txt, 'r') as all_data:
        for line in all_data:
            if len(line) > 5:
                tempo.append(line)
    size = len(tempo)//100
    train = tempo[0:80*size]
    valid = tempo[80*size:90*size]
    test = tempo[90*size:100*size]
    save = open('./data/train/BSY_train_%s.txt' % embedding_file_name, 'w')
    for line in train:
        print(line, file=save)

    save = open('./data/train/BSY_valid_%s.txt' % embedding_file_name, 'w')
    for line in valid:
        print(line, file=save)

    save = open('./data/train/BSY_test_%s.txt' % embedding_file_name, 'w')
    for line in test:
        print(line, file=save)
'''

'''
def make_dataset():
    tempo = []
    with open('./dataset/BSY_all_data.txt', 'r') as all_data:
        for line in all_data:
            if len(line) > 5:
                tempo.append(line)
    size = len(tempo)//100
    train = tempo[0:80*size]
    valid = tempo[80*size:90*size]
    test = tempo[90*size:100*size]
    save = open('./data/train/BSY_train.txt', 'w')
    for line in train:
        print(line, file=save)

    save = open('./data/train/BSY_valid.txt', 'w')
    for line in valid:
        print(line, file=save)

    save = open('./data/train/BSY_test.txt', 'w')
    for line in test:
        print(line, file=save)
'''

def checker():
    print("checker 실행...")
    mal_cnt = 0
    ben_cnt = 0
    #opcode_path = './dataset/opcode/'
    #file_list = os.listdir(opcode_path)
    with open('./dataset/BSY_all_data.txt', 'r') as all_data:
        for line in all_data:
            label = line[0]
            #print(label)
            if label == '0':
                ben_cnt += 1
            elif label == '1':
                mal_cnt += 1
    print("checker... malware count : " + str(mal_cnt))
    print("checker... benign count : " + str(ben_cnt))


def make_dataset():
    mal_cnt = 0
    ben_cnt = 0
    train_file = open('./data/train/JJU1_train.txt', 'w', encoding='utf-8')
    valid_file = open('./data/train/JJU1_valid.txt', 'w', encoding='utf-8')
    test_file = open('./data/train/JJU1_test.txt', 'w', encoding='utf-8')
    with open('./dataset/JJU1_all_data.txt', 'r', encoding='utf-8') as all_data:
        for line in all_data:
            label, _ = line.split('##', 1)
            if label == '0':
                ben_cnt += 1
                if ben_cnt <= 12865:
                    print(line, file=train_file)
                elif ben_cnt > 12865 and ben_cnt <= 14473:
                    print(line, file=valid_file)
                else:
                    print(line, file=test_file)
            elif label == '1':
                mal_cnt += 1
                if mal_cnt <= 17669:
                    print(line, file=train_file)
                elif mal_cnt > 17669 and mal_cnt <= 19877:
                    print(line, file=valid_file)
                else:
                    print(line, file=test_file)

    print("Finish make_dataset... malware cnt : " + str(mal_cnt))
    print("Finish make_dataset... benign cnt : " + str(ben_cnt))


def all_data_mapping(label_file):
    # all_data를 먼저 label, 이름으로 매핑한다.
    all_data_element = open('./labelMap/BSY_all_data_labelMap.txt', 'a', encoding='utf-8')
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)

    for line in csv_reader:  # line[0] = name, line[1] = label
        try:
            print(line[1])
            op_file = open('./dataset/BSY_opcode/%s.txt' % line[0], 'r', encoding='utf-8')    # 있는 dataset인지 확인하기 위해...
            all_data_element.write(str(line[1]) + '##' + str(line[0]) + '\n')
            op_file.close()
        except:
            print(str(line[0]) + " 없어=================================================")
            pass

    csv_data.close()


# dataset의 순서와 각 sample 이름과 label 매핑
def dataset_label_mapping():
    mal_cnt = 0
    ben_cnt = 0
    train_file = open('./labelMap/BSY_train_labelMap.txt', 'w', encoding='utf-8')
    valid_file = open('./labelMap/BSY_valid_labelMap.txt', 'w', encoding='utf-8')
    test_file = open('./labelMap/BSY_test_labelMap.txt', 'w', encoding='utf-8')
    with open('./labelMap/BSY_all_data_labelMap.txt', 'r', encoding='utf-8') as all_data:
        for line in all_data:
            label, name = line.split('##', 1)
            if label == '0':
                ben_cnt += 1
                if ben_cnt <= 11917:
                    print(line, file=train_file)
                elif ben_cnt > 11917 and ben_cnt <= 13407:
                    print(line, file=valid_file)
                else:
                    print(line, file=test_file)
            elif label == '1':
                mal_cnt += 1
                if mal_cnt <= 13099:
                    print(line, file=train_file)
                elif mal_cnt > 13099 and mal_cnt <= 14737:
                    print(line, file=valid_file)
                else:
                    print(line, file=test_file)

    print("Finish make_dataset... malware cnt : " + str(mal_cnt))
    print("Finish make_dataset... benign cnt : " + str(ben_cnt))


def overlap_classifier():
    opcode_file_dir = './dataset/JJU1_opcode/'
    dir = os.listdir(opcode_file_dir)

    for file in dir:
        try:
            file_dir = os.path.join(opcode_file_dir, file)

            with open(file_dir, 'r') as f:
                first_check = False
                delete_check = False
                i = 0
                for line in f:
                    i += 1
                    if first_check == False:
                        first_word = line
                        first_check = True    # 이제 다시는 안들어가게

                    if first_word != line:
                        delete_check = True

                    if i >= 2000:
                        break

            if delete_check == False:
                print(file_dir)
                os.remove(file_dir)

        except:
            pass


def opcode_overlap_delete():
    origin_file_dir = './dataset/opcode_copy1/'
    opcode_file_dir = './dataset/JJU1_opcode/'

    origin_dir = os.listdir(origin_file_dir)          # ['aaa1.txt', 'aaa2.txt', ...]

    for file in origin_dir:
        try:
            origin_file = os.path.join(origin_file_dir, file)
            opcode_file = os.path.join(opcode_file_dir, file)
            with open(origin_file, 'r', encoding='utf-8') as fr:
                first_check = False
                with open(opcode_file, 'w', encoding='utf-8') as fw:
                    for line in fr:
                        if first_check == False:
                            first_word = line
                            first_check = True      # 첫번 째 단어를 저장하고 나면 다시 첫 단어를 저장 안함
                            fw.write(line)          # 첫번 째 단어를 확인하고 나면 적을 수 있도록

                        if first_word != line:      # 체크한 단어와 다음 단어가 같지 않으면
                            first_word = line
                            fw.write(line)
                        else:                       # first word와 line이 같다면 pass
                            pass

        except:
            print(str(file) + ' error!')
            pass


def number_counter():
    num_txt = open('./dataset/BSY_all_data_opcode_cnt.txt', 'r')
    lines = num_txt.readlines()
    num_list = []
    for line in lines:
        k = line.split(" ")
        num_list.append(int(k[4].strip()))
    sorted_list = sorted(num_list)
    print(sorted_list)
    aa = np.arange(0, 620400, 1)
    under1, under2, under3, under4, under5, under6, under7, under8, under9, under10, under11, under12 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in aa:
        #print("%d 개 opcode를 가진 파일의 개수 : %d" % (i + 1, sorted_list.count(i + 1)))
        if i <= 300:
            under1 += sorted_list.count(i + 1)
        elif i > 300 and i <= 600:
            under2 += sorted_list.count(i + 1)
        elif i > 600 and i <= 900:
            under3 += sorted_list.count(i + 1)
        elif i > 900 and i <= 1200:
            under4 += sorted_list.count(i + 1)
        elif i > 1200 and i <= 1500:
            under5 += sorted_list.count(i + 1)
        elif i > 1500 and i <= 1800:
            under6 += sorted_list.count(i + 1)
        elif i > 1800 and i <= 2100:
            under7 += sorted_list.count(i + 1)
        elif i > 2100 and i <= 2400:
            under8 += sorted_list.count(i + 1)
        elif i > 2400 and i <= 2700:
            under9 += sorted_list.count(i + 1)
        elif i > 2700 and i <= 3000:
            under10 += sorted_list.count(i + 1)
        elif i > 3000 and i <= 3300:
            under11 += sorted_list.count(i + 1)
        elif i > 3300 and i <= 3600:
            under12 += sorted_list.count(i + 1)
    print("under1 = %d" % under1)
    print("under2 = %d" % under2)
    print("under3 = %d" % under3)
    print("under4 = %d" % under4)
    print("under5 = %d" % under5)
    print("under6 = %d" % under6)
    print("under7 = %d" % under7)
    print("under8 = %d" % under8)
    print("under9 = %d" % under9)
    print("under10 = %d" % under10)
    print("under11 = %d" % under11)
    print("under12 = %d" % under12)
    under_list = [under1, under2, under3, under4, under5, under6, under7, under8, under9, under10, under11, under12, 6262]
    bins = np.arange(0, 4300, 100)
    hist, bins = np.histogram(under_list, bins)
    plt.hist(sorted_list, bins, rwidth=0.5)
    plt.xlabel('length')
    plt.ylabel('cnt')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


def sequence_calculator():
    # get opcode sequence count
    num_txt = open('./dataset/BSY_all_data_opcode_cnt.txt', 'r')
    lines = num_txt.readlines()
    num_list, file_name_list = [], []
    for line in lines:
        k = line.split(" ")
        file_name_list.append(k[0].strip())
        num_list.append(int(k[4].strip()))
    #print(file_name_list)
    #print(num_list)

    TF_list = []
    sigma_Op_count = sum(num_list)
    #print("Sigma of Opcode sequence length count = %d" % sigma_Op_count)
    for i in num_list:
        TF_value = (i / sigma_Op_count)
        #print("TF_value = %.10f" % TF_value)
        TF_list.append(float(TF_value))
    #print(TF_list)

    return num_list, file_name_list, TF_list


def sequence_zero_Detector():
    zero_file_list = './dataset/entropy_zero/'
    zero_file_name = os.listdir(zero_file_list)
    name_list = []
    for i in zero_file_name:
        modified = i.strip().replace(".txt", "")
        name_list.append(modified)
        print(modified)
    print(name_list)

    entropy_zero_csv = open('./zero_entropy_list.csv', 'w', encoding='utf-8')
    zero_writer = csv.writer(entropy_zero_csv)
    csv_file = open('./Total_File_Infor.csv', 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_file)
    for file in csv_reader:
        try:
            if file[0] in name_list:
                print(file[0] + " start...")
                zero_writer.writerow(file)

        except:
            print("error!")
            pass


def overlap_delete_EntropyCheck():
    Entropy_threshold = 7.2

    label_file = './Total_File_Infor.csv'
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)
    origin_file_dir = './dataset/opcode_copy2/'
    opcode_file_dir = './dataset/JJU4_opcode/'
    print(opcode_file_dir + "start!!")
    print('Entropy threshold = %f' % Entropy_threshold)

    for file in csv_reader:
        try:
            if float(file[3]) < Entropy_threshold:
                print("file[3] = %f" % float(file[3]))
                origin_file = os.path.join(origin_file_dir, file[0] + '.txt')
                opcode_file = os.path.join(opcode_file_dir, file[0] + '.txt')

                with open(origin_file, 'r', encoding='utf-8') as fr:
                    first_check = False
                    with open(opcode_file, 'w', encoding='utf-8') as fw:
                        for line in fr:
                            if first_check == False:
                                first_word = line
                                first_check = True      # 첫번 째 단어를 저장하고 나면 다시 첫 단어를 저장 안함
                                fw.write(line)          # 첫번 째 단어를 확인하고 나면 적을 수 있도록

                            if first_word != line:      # 체크한 단어와 다음 단어가 같지 않으면
                                first_word = line
                                fw.write(line)
                            else:                       # first word와 line이 같다면 pass
                                pass
            else:
                print('file name = %s      Entropy = %f' % (file[0], float(file[3])))
                pass
        except:
            pass


def main():
    benign_cnt = 0
    malware_cnt = 0

    train_label = './data/label/trainSet.csv'
    pre_label = './data/label/preSet.csv'
    final1_label = './data/label/finalSet1.csv'
    final2_label = './data/label/finalSet2.csv'

    word_vocab = './data/train/JJU1_word_vocab.txt'

    w2v100_model_dir = './embedding_model/JJU1_word2vec_model_100'
    w2v200_model_dir = './embedding_model/JJU1_word2vec_model_200'
    w2v300_model_dir = './embedding_model/JJU1_word2vec_model_300'
    fast100_model_dir = './embedding_model/JJU1_fasttext_model_100'
    fast200_model_dir = './embedding_model/JJU1_fasttext_model_200'
    fast300_model_dir = './embedding_model/JJU1_fasttext_model_300'

    # print("start all_dataMaker...")
    #JJU_dataMaker(train_label)
    #JJU_dataMaker(pre_label)
    #JJU_dataMaker(final1_label)
    #JJU_dataMaker(final2_label)
    #all_dataMaker(train_label)
    #all_dataMaker(pre_label)
    #all_dataMaker(final1_label)
    #all_dataMaker(final1_label
    #all_dataMaker(final2_label)
    #print("finish all_dataMaker...")

    #JJU2_dataMaker('./Total_File_Infor.csv')
    # print("finish all_dataMaker...")


    # train embedding file
    train_embedding()
    w2v100_model = word2vec.Word2Vec.load(w2v100_model_dir)
    build_vocab(w2v100_model, word_vocab)

def predict_name_mapping(y_pred):
    predict_cnt = 0
    name_list, benign_list = [], []
    with open('./labelMap/BSY_test_labelMap.txt', 'r', encoding='utf-8') as testMap:
        for line in testMap:
            if line is '\n':
                pass
            else:
                line_strip = line.strip()
                label, name = line_strip.split('##', 1)
                if name.find("\ufeff") != -1:
                    name = name.replace("\ufeff", "")
                name_list.append(name)
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            benign_list.append(name_list[i])
    print(benign_list)

    return benign_list



#overlap_delete_EntropyCheck()
#opcode_overlap_delete()

#main()
#checker()
#sequence_zero_Detector()
#make_dataset()
#number_counter()
#IG_calculator()

# data mapping
'''train_label = './data/label/trainSet.csv'
pre_label = './data/label/preSet.csv'
final1_label = './data/label/finalSet1.csv'
final2_label = './data/label/finalSet2.csv'

#all_data_mapping(train_label)
#all_data_mapping(pre_label)
##all_data_mapping(final1_label)
#all_data_mapping(final2_label)

dataset_label_mapping()'''

#predict_name_mapping()

train_embedding()