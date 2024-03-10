import numpy as np
import csv
import os
import math
import shutil


# 각 항목들을 0.1 ~ 1 값으로 scaling 해주는 함수
def dict_MinMax_scaler(dict, min_Value, max_Value):
    norm_dict = {}
    for key, value in dict.items():
        scaled_value = 0.9 * ((value - min_Value) / (max_Value - min_Value)) + 0.1
        norm_dict['%s' % key] = scaled_value

    return norm_dict


def tensor_processing(file_name, TENSOR_SIZE, protect_dict, type_dict, size_dict, category_dict, api_dict):
    f = open(file_name, 'r', encoding='utf-8')
    csv_reader = csv.reader(f)

    regions_arr = np.zeros((TENSOR_SIZE, TENSOR_SIZE))
    category_arr = np.zeros((TENSOR_SIZE, TENSOR_SIZE))
    api_arr = np.zeros((TENSOR_SIZE, TENSOR_SIZE))

    # dim1_low_cnt1, dim1_low_cnt2, dim1_low_cnt3 = protect_arr start, type_arr start, size_arr start
    protect_low_cnt, type_low_cnt, size_low_cnt = 0, int(TENSOR_SIZE / 3), int(2 * (TENSOR_SIZE / 3))
    regions_col_cnt = 0
    behavior_low_cnt = 0
    behavior_col_cnt = 0

    regions_Full, behavior_Full = False, False
    regions_cnt, behavior_cnt = 0, 0

    for line in csv_reader:
        # channel 1 processing
        if line[0] == 'procMemory':
            if line[1] == 'regions' and regions_Full is False:
                # protect processing
                protect = line[2].strip()
                protect_vector = protect_dict[protect]
                regions_arr[protect_low_cnt][regions_col_cnt] = protect_vector

                # type processing
                type = line[6].strip()
                type_vector = type_dict[type]
                regions_arr[type_low_cnt][regions_col_cnt] = type_vector

                # size processing
                size = line[7].strip()
                size_vector = size_dict[size]
                regions_arr[size_low_cnt][regions_col_cnt] = size_vector

                regions_col_cnt += 1

                if regions_col_cnt == TENSOR_SIZE:
                    regions_col_cnt = 0

                    protect_low_cnt += 1
                    type_low_cnt += 1
                    size_low_cnt += 1

                    if size_low_cnt == (int(TENSOR_SIZE / 3) + int(2 * (TENSOR_SIZE / 3))):    # regions를 전부 채웠을 때 224x224일 땐 150 and 512x512일 땐 342
                        regions_Full = True

        # channel 2 and 3 processing
        elif line[0] == 'behavior':
            if line[1] == 'calls' and behavior_Full is False:
                # category ch2 processing
                category = line[2].strip()
                category_vector = category_dict[category]
                category_arr[behavior_low_cnt][behavior_col_cnt] = category_vector

                # api ch3 processing
                api = line[3].strip()
                api_vector = api_dict[api]
                api_arr[behavior_low_cnt][behavior_col_cnt] = api_vector

                behavior_col_cnt += 1

                if behavior_col_cnt == TENSOR_SIZE:
                    behavior_col_cnt = 0

                    behavior_low_cnt += 1

                    if behavior_low_cnt == TENSOR_SIZE:    # regions를 전부 채웠을 때 224x224일 땐 150 and 512x512일 땐 342
                        behavior_Full = True


    np_file = np.dstack([regions_arr, category_arr, api_arr])
    #print(np_file)

    return np_file


def dict_maker(txt_file_path):
    frequency_txt = open(txt_file_path, 'r', encoding='utf-8')
    frequency_dict = {}

    for txt_line in frequency_txt:
        txt = txt_line.strip()
        txt = txt.split('#', 1)
        name = txt[0]
        frequency = int(txt[1])

        frequency_dict.setdefault(name, frequency)
    min_Value = min(frequency_dict.values())
    max_Value = max(frequency_dict.values())

    frequency_norm_dict = dict_MinMax_scaler(frequency_dict, min_Value, max_Value)

    return frequency_norm_dict
'''

if __name__ == "__main__":
    #TENSOR_SIZE1 = 224
    #TENSOR_SIZE2 = 336
    TENSOR_SIZE3 = 448

    protect_frequency = './data/frequency/protect_frequency.txt'
    type_frequency = './data/frequency/type_frequency.txt'
    size_frequency = './data/frequency/size_frequency.txt'
    category_frequency = './data/frequency/category_frequency.txt'
    api_frequency = './data/frequency/api_frequency.txt'

    # get normalized dictionary
    protect_dict = dict_maker(protect_frequency)
    type_dict = dict_maker(type_frequency)
    size_dict = dict_maker(size_frequency)
    category_dict = dict_maker(category_frequency)
    api_dict = dict_maker(api_frequency)

    foler_path = './data/csv/'  # csv file's path
    csv_files = os.listdir(foler_path)    # csv file list
    filecnt = 0
    for file_name in csv_files:
        print(file_name)

        file_path = os.path.join(foler_path, file_name)    # join the csv file name and whole path

        #np_file224 = tensor_processing(file_path, TENSOR_SIZE1, protect_dict, type_dict, size_dict, category_dict, api_dict)
        #np.save('./data/npy/%d_size/%s' % (TENSOR_SIZE1, file_name), np_file224)

        #np_file384 = tensor_processing(file_path, TENSOR_SIZE2, protect_dict, type_dict, size_dict, category_dict, api_dict)
        #np.save('./data/npy/%d_size/%s' % (TENSOR_SIZE2, file_name), np_file384)

        np_file512 = tensor_processing(file_path, TENSOR_SIZE3, protect_dict, type_dict, size_dict, category_dict, api_dict)
        np.save('./data/npy/%d_size/%s' % (TENSOR_SIZE3, file_name), np_file512)

        filecnt += 1
        print("file " + str(filecnt) + " / 38905")


######################################################################################################
# train, valid, test set split part
######################################################################################################
def dataset_classifier(train_ID, train_Label, valid_ID, valid_Label, test_ID, test_Label, image_size):
    img_dir = os.listdir('./data/npy/%d_size/' % image_size)
    src = './data/npy/%d_size/' % image_size
    train_dst = './data/npy/train%d/' % image_size
    valid_dst = './data/npy/valid%d/' % image_size
    test_dst = './data/npy/test%d/' % image_size

    for img_name in img_dir:
        origin_img_name = img_name
        img_name = img_name.strip()
        if img_name.find(".csv") != -1:
            img_name = img_name.replace(".csv", "")

        if img_name.find(".vir") != -1:
            img_name = img_name.replace(".vir", "")

        img_name = img_name.replace(".npy", ".vir")

        if (img_name not in test_ID) and (img_name in train_ID):
            shutil.move(src + origin_img_name, train_dst + img_name + '.npy')
        elif (img_name not in test_ID) and (img_name in valid_ID):
            shutil.move(src + origin_img_name, valid_dst + img_name + '.npy')
        elif img_name in test_ID:
            shutil.move(src + origin_img_name, test_dst + img_name + '.npy')


def getData(path):
    getFile = open(path, 'r', encoding='utf-8')
    getData = csv.reader(getFile)
    ID_list, Label_list = [], []

    for data in getData:
        name = data[0]
        label = data[1]
        if name.find("\ufeff") != -1:
            name = name.replace("\ufeff", "")
        ID_list.append(name)
        Label_list.append(label)
    getFile.close()

    return ID_list, Label_list


# data preprocessing
train_ID, train_Label = getData('./data/labels/train_ID_Labels.csv')
valid_ID, valid_Label = getData('./data/labels/valid_ID_Labels.csv')
test_ID, test_Label = getData('./data/labels/test_ID_Labels.csv')
print("fucking god")
dataset_classifier(train_ID, train_Label, valid_ID, valid_Label, test_ID, test_Label, image_size=224)
dataset_classifier(train_ID, train_Label, valid_ID, valid_Label, test_ID, test_Label, image_size=336)
dataset_classifier(train_ID, train_Label, valid_ID, valid_Label, test_ID, test_Label, image_size=448)
'''
'''
data1 = np.load('./data/npy/448_size/0000f70b7dc9b8009beb8d9500993b7d.csv.npy')
print("data1")
print(data1)
print(data1[0])
print(data1[1])
print(data1[2])
data2 = np.load('./data/npy/448_size/000cf7d1523400b8304ea515f5447dc7.csv.npy')
print("data2")
print(data2[0])
print(data2[1])
print(data2[2])
data3 = np.load('./data/npy/448_size/000e0e4377b576f422b3b93200a3452f.csv.npy')
print("data3")
print(data3[0])
print(data3[1])
print(data3[2])
'''
data4 = np.load('./data/npy/448_size/000e7bda840c2712a3db202a7b0e6bc2.csv.npy')
print("data4")
print(data4[0])
print(data4[1])
print(data4[2])
with open('./region_npy_to_csv.csv', 'w', newline="", encoding="utf-8") as f:
    csv_wr = csv.writer(f)
    # region
    for i in range(448):
        reg = []
        for j in range(448):
            reg.append(data4[i][j][0])
        csv_wr.writerow(reg)
        del reg

with open('./category_npy_to_csv.csv', 'w', newline="", encoding="utf-8") as f:
    csv_wr = csv.writer(f)
    # cate
    for i in range(448):
        cate = []
        for j in range(448):
            cate.append(data4[i][j][1])
        csv_wr.writerow(cate)
        del cate

with open('./api_npy_to_csv.csv', 'w', newline="", encoding="utf-8") as f:
    csv_wr = csv.writer(f)
    # api
    for i in range(448):
        api = []
        for j in range(448):
            api.append(data4[i][j][2])
        csv_wr.writerow(api)
        del api