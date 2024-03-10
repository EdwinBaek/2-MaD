import numpy as np
import csv
import os
import math
import shutil


'''
if __name__ == "__main__":
    train_files = os.listdir('./data/npy/train448/')

    for train_npy in train_files:
        file = np.load('./data/npy/train448/' + train_npy)
        ZeroChecker = np.nonzero(file)
        ZeroSum = np.sum(ZeroChecker)
        if ZeroSum == 0:
            print("영행렬 발견!")
        else:
            shutil.copyfile('./data/npy/train448/' + train_npy, './data/npy/train448_deleted/' + train_npy)


    valid_files = os.listdir('./data/npy/valid448/')
    for valid_npy in valid_files:
        file = np.load('./data/npy/valid448/' + valid_npy)
        ZeroChecker = np.nonzero(file)
        ZeroSum = np.sum(ZeroChecker)
        if ZeroSum == 0:
            print("영행렬 발견!")
        else:
            shutil.copyfile('./data/npy/valid448/' + valid_npy, './data/npy/valid448_deleted/' + valid_npy)

'''
'''
# train, valid, test set csv maker
def dataset_classifier():
    train_list, valid_list, test_list = [], [], []
    train_label, valid_label, test_label = [], [], []
    test_cnt = 0

    # testset list 만들기
    with open('./labelMap/BSY_test_labelMap.txt', 'r', encoding='utf-8') as testMap:
        for line in testMap:
            if line == '\n':
                pass
            else:
                line_strip = line.strip()
                label, name = line_strip.split('##', 1)
                if name.find("\ufeff") != -1:
                    name = name.replace("\ufeff", "")
                test_list.append(name)
                test_label.append(label)
                test_cnt += 1

    print("test_cnt")
    print(test_cnt)


    # train, valid list 만들기
    all_label_csv = open('./data/labels/RealAllLabel.csv', 'r', encoding='utf-8')
    csv_rdr = csv.reader(all_label_csv)
    malware_cnt, benign_cnt = 0, 0
    train_malware_cnt, train_benign_cnt, valid_malware_cnt, valid_benign_cnt = 0, 0, 0, 0
    for csvLine in csv_rdr:
        if csvLine[0] not in test_list:
            if csvLine[1] == '0':
                benign_cnt += 1
                if benign_cnt % 8 != 0:
                    train_list.append(csvLine[0])
                    train_label.append(csvLine[1])
                    train_benign_cnt += 1
                else:
                    valid_list.append(csvLine[0])
                    valid_label.append(csvLine[1])
                    valid_benign_cnt += 1
            elif csvLine[1] == '1':
                malware_cnt += 1
                if malware_cnt % 8 != 0:
                    train_list.append(csvLine[0])
                    train_label.append(csvLine[1])
                    train_malware_cnt += 1
                else:
                    valid_list.append(csvLine[0])
                    valid_label.append(csvLine[1])
                    valid_malware_cnt += 1
    print('malware_cnt')
    print(malware_cnt)
    print('benign_cnt')
    print(benign_cnt)
    print('train_malware_cnt')
    print(train_malware_cnt)
    print('train_benign_cnt')
    print(train_benign_cnt)
    print('valid_malware_cnt')
    print(valid_malware_cnt)
    print('valid_benign_cnt')
    print(valid_benign_cnt)

    train_dir = os.listdir('./data/npy/train448_deleted')
    valid_dir = os.listdir('./data/npy/valid448_deleted')

    train_csv = open('./data/labels/deleted_train_ID_Labels.csv', 'w', newline='', encoding='utf-8')
    train_wr = csv.writer(train_csv)
    valid_csv = open('./data/labels/deleted_valid_ID_Labels.csv', 'w', newline='', encoding='utf-8')
    valid_wr = csv.writer(valid_csv)
    #test_csv = open('./data/labels/test_ID_Labels.csv', 'a', encoding='utf-8')
    #test_wr = csv.writer(test_csv)
    print("train start")
    for img_name in train_dir:
        img_name = img_name.strip()
        img_name = img_name.replace(".npy", "")

        train_label_index = train_list.index(img_name)
        train_wr.writerow([img_name, train_label[train_label_index]])

    print("valid start")
    for img_name in valid_dir:
        img_name = img_name.strip()
        img_name = img_name.replace(".npy", "")

        valid_label_index = valid_list.index(img_name)
        valid_wr.writerow([img_name, valid_label[valid_label_index]])

print("come on babe make csv file of deleted file")
dataset_classifier()
'''

def log_noise_remover(filename, csv_file):
    print("%s" % filename)
    new_log = open("./logs/%s.csv" % filename, 'w', newline='', encoding='utf-8')
    new_log_wr = csv.writer(new_log)
    i = 0
    with open(csv_file, 'r', encoding='utf-8') as f:
        for line in f:
            if i == 0:
                new_log_wr.writerow(['epoch', 'train_acc', 'train_loss', 'valid_acc', 'valid_loss'])
                i += 1
            else:
                line = line.strip()
                line = line.split(',')
                epoch = line[0]
                train_acc, train_loss = line[1].replace('"tensor(', ''), line[3].replace('"tensor(', '')
                valid_acc, valid_loss = line[6].replace('"tensor(', ''), line[8].replace('"tensor(', '')

                new_log_wr.writerow([epoch, train_acc, train_loss, valid_acc, valid_loss])
'''
log_noise_remover(filename='new_VGG16_224', csv_file='./logs/VGG16_224_logs.csv')
log_noise_remover(filename='new_VGG16_336', csv_file='./logs/VGG16_336_logs.csv')
log_noise_remover(filename='new_EfficientNet_224', csv_file='./logs/EfficientNet_224_logs.csv')
log_noise_remover(filename='new_EfficientNet_336', csv_file='./logs/EfficientNet_336_logs.csv')
log_noise_remover(filename='new_EfficientNet_448', csv_file='./logs/EfficientNet_448_logs.csv')
log_noise_remover(filename='new_ResNet50_224', csv_file='./logs/ResNet50_224_logs.csv')
log_noise_remover(filename='new_ResNet50_336', csv_file='./logs/ResNet50_336_logs.csv')
log_noise_remover(filename='new_ResNet50_448', csv_file='./logs/ResNet50_448_logs.csv')
log_noise_remover(filename='new_ResNeXt50_224', csv_file='./logs/ResNeXt50_224_logs.csv')
log_noise_remover(filename='new_ResNeXt50_336', csv_file='./logs/ResNeXt50_336_logs.csv')
log_noise_remover(filename='new_ResNeXt50_448', csv_file='./logs/ResNeXt50_448_logs.csv')
'''