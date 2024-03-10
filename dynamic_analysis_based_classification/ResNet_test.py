import csv
import time
import math
import scipy.io
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import shutil


import torch
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(('Available devices ', torch.cuda.device_count()))
print('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


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
        Label_list.append(float(label))
    getFile.close()

    return ID_list, Label_list


# Transform, Data Set ready
class MyDataset(Dataset):
    def __init__(self, image_path, mode, transform=None):
        #test_ID, test_Label = getData('./data/labels/benign_test_ID_Labels.csv')    # 전체 모델에 대해
        test_ID, test_Label = getData('./data/labels/test_ID_Labels.csv')     # CNN만 돌릴때

        self.labels = np.array(test_Label, dtype=float)
        self.images = [image_path + '%s.npy' % i for i in test_ID]

        self.transform = transform

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(np.load(image))
        return image, label

    def __len__(self):
        return len(self.labels)


# benign으로 판별된 txt 파일이 있을 때 사용 가능
def predict_to_benign_dataset_classifier():
    benign_test_path = './data/npy/benign_test_448/'
    test_path = './data/npy/test448/'
    with open('./data/predict_to_benign.txt', 'r', encoding='utf-8') as benign_list:
        for i in benign_list:
            try:
                benign_name = i.strip()
                shutil.copyfile(test_path + benign_name + '.npy', benign_test_path + benign_name + '.npy')
                print(benign_name)
            except:
                print("error! error! about " + i)
                pass

def label_maker():
    benign_test_ID_label = open('./data/labels/benign_test_ID_Labels.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(benign_test_ID_label)

    test_ID, test_Label = getData('./data/labels/test_ID_Labels.csv')

    with open('./data/predict_to_benign.txt', 'r', encoding='utf-8') as benign_list:
        for i in benign_list:
            try:
                benign_name = i.strip()
                ind = test_ID.index(benign_name)
                label = int(test_Label[int(ind)])
                writer.writerow([benign_name, label])
            except:
                print("error! error! about " + i)
                pass

def testset_missing_finder():
    origin = []
    with open('./data/labelMap/BSY_test_labelMap.txt', 'r', encoding='utf-8') as o:
        for line in o:
            line = line.strip()
            if line != '':
                label, name = line.split('##', 1)
                origin.append(name)


    csv_name = []     # 1472개 missing이 발생한 부분
    with open('./data/labels/test_ID_Labels.csv', 'r', encoding='utf-8') as c:
        for line in c:
            name, label = line.split(',')
            csv_name.append(name)
    '''
    txt_name = []     # 1475개
    with open('./data/predict_to_benign.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            txt_name.append(line)

    '''
    for i in origin:
        if i not in csv_name:
            print("find missing one")
            print(i)



# ResNet152 modeling
def conv_start():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=4),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )


def bottleneck_block(in_dim, mid_dim, out_dim, down=False):
    layers = []
    if down:
        layers.append(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=2, padding=0))
    else:
        layers.append(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0))
    layers.extend([
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_dim, out_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_dim),
    ])
    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, down:bool = False, starting:bool=False) -> None:
        super(Bottleneck, self).__init__()
        if starting:
            down = False
        self.block = bottleneck_block(in_dim, mid_dim, out_dim, down=down)
        self.relu = nn.ReLU(inplace=True)
        if down:
            self.changedim = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2, padding=0),
                                           nn.BatchNorm2d(out_dim))
        else:
            self.changedim = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm2d(out_dim))


    def forward(self, x):
        identity = self.changedim(x)
        x = self.block(x)
        x += identity
        x = self.relu(x)
        return x


def make_layer(in_dim, mid_dim, out_dim, repeats, starting=False):
    layers = []
    layers.append(Bottleneck(in_dim, mid_dim, out_dim, down=True, starting=starting))
    for _ in range(1, repeats):
        layers.append(Bottleneck(out_dim, mid_dim, out_dim, down=False))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, repeats: list = [3, 4, 6, 3], num_classes=1000):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        # 1번
        self.conv1 = conv_start()

        # 2번
        base_dim = 64
        self.conv2 = make_layer(base_dim, base_dim, base_dim * 4, repeats[0], starting=True)
        self.conv3 = make_layer(base_dim * 4, base_dim * 2, base_dim * 8, repeats[1])
        self.conv4 = make_layer(base_dim * 8, base_dim * 4, base_dim * 16, repeats[2])
        self.conv5 = make_layer(base_dim * 16, base_dim * 8, base_dim * 32, repeats[3])

        # 3번
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifer = nn.Linear(2048 * 8 * 8, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        # 3번 2048x1 -> 1x2048
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        return x

if __name__ == '__main__':
    #predict_to_benign_dataset_classifier()
    #label_maker()
    #testset_missing_finder()

    # Hyper-Parameters
    IMAGE_SIZE = 448
    #MODEL_PATH = 'ResNet50_224_checkpoint_8epoch_0.8772_0.2841.pth'
    #MODEL_PATH = 'ResNet50_336_checkpoint_15epoch_0.8816_0.2796.pth'
    MODEL_PATH = 'ResNet50_448_checkpoint_14epoch_0.8793_0.2808.pth'
    #test_Path = './data/npy/benign_test_%s/' % IMAGE_SIZE    # 전체 모델에 대해
    test_Path = './data/npy/test%s/' % IMAGE_SIZE     # CNN만 돌릴 땨
    num_classes = 2
    BATCH = 8

    print("%d size %s model test start!" % (IMAGE_SIZE, MODEL_PATH))
    test_value = []

    # load Dataset
    test_dataset = MyDataset(test_Path,
                             mode='test',
                             transform=transforms.Compose([transforms.ToTensor()]))

    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)
    print('Test size:', len(test_loader))

    # load torch model
    model = ResNet(num_classes=num_classes)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.99))

    checkpoint = torch.load('./model/%s' % MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #summary(model, (3, IMAGE_SIZE, IMAGE_SIZE))
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    classes = ['Benign', 'Malware']

    # Test part
    model.eval()
    test_pred = torch.LongTensor()
    total_loss = 0

    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.int64)

            output = model(data)
            loss = criterion(output, label)

            #pred = (output.argmax(dim=1) == label).float().mean()
            pred = output.cpu().data.max(1, keepdim=True)[1]
            test_pred = torch.cat((test_pred, pred), dim=0)
            total_loss += loss.item()

    out_df = pd.DataFrame(np.c_[np.arange(1, len(test_dataset) + 1)[:, None], test_pred.numpy()],
                          columns=['ImageID', 'Label'])
    print(out_df.head())
    out_df.to_csv('./data/%s_submission.csv' % MODEL_PATH, index=False)

    #_, test_Label = getData('./data/labels/benign_test_ID_Labels.csv')    # 전체 모델일 때
    _, test_Label = getData('./data/labels/test_ID_Labels.csv')     # 그냥 CNN만 썼을 때
    test_Label = list(map(int, test_Label))

    y_true = test_Label
    y_pred = test_pred.numpy()
    loss = total_loss / len(test_loader.dataset)

    confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print('Test value tp : ' + str(tp))
    print('Test value fp : ' + str(fp))
    print('Test value fn : ' + str(fn))
    print('Test value tn : ' + str(tn))

    print('\nTest loss >>>>> (%.4f%%)' % (loss))
    print('\nTest accuracy >>>>> (%.4f%%)' % (accuracy_score(y_true, y_pred)))
    test_value.append([loss, accuracy_score(y_true, y_pred), tp, fn, fp, fn])

    print(time.time() - start_time)
    print("%d size %s model TEST Finished!" % (IMAGE_SIZE, MODEL_PATH))


