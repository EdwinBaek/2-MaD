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



class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

def _RoundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c

def _RoundRepeats(r):
    return int(math.ceil(r))

def _DropPath(x, drop_prob, training):
    if drop_prob > 0 and training:
        keep_prob = 1 - drop_prob
        if x.is_cuda:
            mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        else:
            mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)

    return x

def _BatchNorm(channels, eps=1e-3, momentum=0.01):
    return nn.BatchNorm2d(channels, eps=eps, momentum=momentum)

def _Conv3x3Bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        _BatchNorm(out_channels),
        Swish()
    )

def _Conv1x1Bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        _BatchNorm(out_channels),
        Swish()
    )

class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = Swish()
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y

        return y

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_path_rate):
        super(MBConvBlock, self).__init__()

        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0.0)
        self.residual_connection = (stride == 1 and in_channels == out_channels)
        self.drop_path_rate = drop_path_rate

        conv = []

        if expand:
            # expansion phase
            pw_expansion = nn.Sequential(
                nn.Conv2d(in_channels, expand_channels, 1, 1, 0, bias=False),
                _BatchNorm(expand_channels),
                Swish()
            )
            conv.append(pw_expansion)

        # depthwise convolution phase
        dw = nn.Sequential(
            nn.Conv2d(
                expand_channels,
                expand_channels,
                kernel_size,
                stride,
                kernel_size//2,
                groups=expand_channels,
                bias=False
            ),
            _BatchNorm(expand_channels),
            Swish()
        )
        conv.append(dw)

        if se:
            # squeeze and excite
            squeeze_excite = SqueezeAndExcite(expand_channels, in_channels, se_ratio)
            conv.append(squeeze_excite)

        # projection phase
        pw_projection = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, 1, 1, 0, bias=False),
            _BatchNorm(out_channels)
        )
        conv.append(pw_projection)

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            return x + _DropPath(self.conv(x), self.drop_path_rate, self.training)
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    config = [
        #(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats)
        [32,  16,  3, 1, 1, 0.25, 1],
        [16,  24,  3, 2, 6, 0.25, 2],
        [24,  40,  5, 2, 6, 0.25, 2],
        [40,  80,  3, 2, 6, 0.25, 3],
        [80,  112, 5, 1, 6, 0.25, 3],
        [112, 192, 5, 2, 6, 0.25, 4],
        [192, 320, 3, 1, 6, 0.25, 1]
    ]

    def __init__(self, param, num_classes=1000, stem_channels=32, feature_size=1280, drop_connect_rate=0.2):
        super(EfficientNet, self).__init__()

        # scaling width
        width_coefficient = param[0]
        if width_coefficient != 1.0:
            stem_channels = _RoundChannels(stem_channels*width_coefficient)
            for conf in self.config:
                conf[0] = _RoundChannels(conf[0]*width_coefficient)
                conf[1] = _RoundChannels(conf[1]*width_coefficient)

        # scaling depth
        depth_coefficient = param[1]
        if depth_coefficient != 1.0:
            for conf in self.config:
                conf[6] = _RoundRepeats(conf[6]*depth_coefficient)

        # scaling resolution
        input_size = param[2]

        # stem convolution
        self.stem_conv = _Conv3x3Bn(3, stem_channels, 2)

        # total #blocks
        total_blocks = 0
        for conf in self.config:
            total_blocks += conf[6]

        # mobile inverted bottleneck
        blocks = []
        for in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats in self.config:
            # drop connect rate based on block index
            drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
            blocks.append(MBConvBlock(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_rate))
            for _ in range(repeats-1):
                drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
                blocks.append(MBConvBlock(out_channels, out_channels, kernel_size, 1, expand_ratio, se_ratio, drop_rate))
        self.blocks = nn.Sequential(*blocks)

        # last several layers
        self.head_conv = _Conv1x1Bn(self.config[-1][1], feature_size)
        #self.avgpool = nn.AvgPool2d(input_size//32, stride=1)
        self.dropout = nn.Dropout(param[3])
        self.classifier = nn.Linear(feature_size, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = torch.mean(x, (2, 3))
        x = self.dropout(x)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



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




if __name__ == '__main__':
    #predict_to_benign_dataset_classifier()
    #label_maker()
    #testset_missing_finder()

    # Hyper-Parameters
    IMAGE_SIZE = 448
    #MODEL_PATH = 'EfficientNet_224_checkpoint_7epoch_0.8709_0.2909.pth'
    #MODEL_PATH = 'EfficientNet_336_checkpoint_8epoch_0.8765_0.2792.pth'
    MODEL_PATH = 'EfficientNet_448_checkpoint_27epoch_0.8923_0.2600.pth'
    #test_Path = './data/npy/benign_test_%s/' % IMAGE_SIZE    # 전체 모델에 대해
    test_Path = './data/npy/test%s/' % IMAGE_SIZE     # CNN만 돌릴 땨
    num_classes = 2
    BATCH = 8

    net_param = {
        # 'efficientnet type': (width_coef, depth_coef, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5)
    }
    param = net_param['efficientnet-b3']
    print("%d size %s model test start!" % (IMAGE_SIZE, MODEL_PATH))
    test_value = []

    # load Dataset
    test_dataset = MyDataset(test_Path,
                             mode='test',
                             transform=transforms.Compose([transforms.ToTensor()]))

    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)
    print('Test size:', len(test_loader))

    # load torch model
    model = EfficientNet(param, num_classes=num_classes)
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


