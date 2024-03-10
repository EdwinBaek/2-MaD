import csv
import time
import scipy.io
import numpy as np
from PIL import Image
import torch

print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(('Available devices ', torch.cuda.device_count()))
print('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))
import math
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Hyper-Parameters
IMAGE_SIZE = 448
MODEL_NAME = 'EfficientNet'
CSVLog_name = './logs/deleted_%s_%s_logs.csv' % (MODEL_NAME, IMAGE_SIZE)
train_Path = './data/npy/train%s_deleted/' % IMAGE_SIZE
valid_Path = './data/npy/valid%s_deleted/' % IMAGE_SIZE
test_Path = './data/npy/test%s/' % IMAGE_SIZE
num_classes = 2
BATCH = 32
EPOCH = 30
train_acc_value, train_loss_value, valid_acc_value, valid_loss_value = 0, 0, 0, 0


def save_Checkpoint(epoch, model, optimizer, filename):
    state = {
        'Epoch': epoch,
        'State_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)


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
        '''
        if label == '1':
            print("1 일떄")
            print(int(label))
            #Label_list.append([0, 1])
            #Label_list.append([0.0, 1.0])
        else:
            print("0 일떄")
            print(int(label))
            #Label_list.append([1, 0])
            #Label_list.append([1.0, 0.0])
        '''
    getFile.close()

    return ID_list, Label_list


# Transform, Data Set ready
class MyDataset(Dataset):
    def __init__(self, image_path, mode, transform=None):
        train_ID, train_Label = getData('./data/labels/deleted_train_ID_Labels.csv')
        valid_ID, valid_Label = getData('./data/labels/deleted_valid_ID_Labels.csv')
        test_ID, test_Label = getData('./data/labels/test_ID_Labels.csv')
        if mode == 'train':
            self.labels = np.array(train_Label, dtype=float)
            self.images = [image_path + '%s.npy' % i for i in train_ID]
        elif mode == 'valid':
            self.labels = np.array(valid_Label, dtype=float)
            # self.labels = self.labels[:, np.newaxis]
            self.images = [image_path + '%s.npy' % i for i in valid_ID]
        else:
            self.labels = np.array(test_Label, dtype=float)
            # self.labels = self.labels[:, np.newaxis]
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


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("Swish")
        return x * self.sigmoid(x)

def _RoundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    #print("_RoundChannels")
    return new_c

def _RoundRepeats(r):
    #print("_RoundRepeats")
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
    #print("_DropPath")
    return x

def _BatchNorm(channels, eps=1e-3, momentum=0.01):
    #print("_BatchNorm")
    return nn.BatchNorm2d(channels, eps=eps, momentum=momentum)

def _Conv3x3Bn(in_channels, out_channels, stride):
    #print("_Conv3x3Bn")
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        _BatchNorm(out_channels),
        Swish()
    )

def _Conv1x1Bn(in_channels, out_channels):
    #print("_Conv1x1Bn")
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        _BatchNorm(out_channels),
        Swish()
    )

class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()
        #print("SqueezeAndExcite")
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
        #print("MBConvBlock")
        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0.0)
        self.residual_connection = (stride == 1 and in_channels == out_channels)
        self.drop_path_rate = drop_path_rate

        print("expand = %s" % expand)
        print("expand_channels = %d" % expand_channels)
        print("se = %s" % se)
        print("residual_connection = %s" % self.residual_connection)
        conv = []

        if expand:
            # expansion phase
            #print("expand")
            pw_expansion = nn.Sequential(
                nn.Conv2d(in_channels, expand_channels, 1, 1, 0, bias=False),
                _BatchNorm(expand_channels),
                Swish()
            )
            conv.append(pw_expansion)

        # depthwise convolution phase
        #print("depthwise")
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
            #print("se")
            # squeeze and excite
            squeeze_excite = SqueezeAndExcite(expand_channels, in_channels, se_ratio)
            conv.append(squeeze_excite)

        # projection phase
        pw_projection = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, 1, 1, 0, bias=False),
            _BatchNorm(out_channels)
        )
        conv.append(pw_projection)
        print("conv")
        print(conv)

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
            print("stem_channels")
            print(stem_channels)
            for conf in self.config:
                conf[0] = _RoundChannels(conf[0]*width_coefficient)
                print("conf[0]")
                print(conf[0])
                conf[1] = _RoundChannels(conf[1]*width_coefficient)
                print("conf[1]")
                print(conf[1])

        # scaling depth
        depth_coefficient = param[1]
        if depth_coefficient != 1.0:
            for conf in self.config:
                conf[6] = _RoundRepeats(conf[6]*depth_coefficient)
                print("conf[6]")
                print(conf[6])

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
        print(x.shape)
        x = self.stem_conv(x)
        print(x.shape)
        x = self.blocks(x)
        print(x.shape)
        x = self.head_conv(x)
        print(x.shape)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = torch.mean(x, (2, 3))
        x = self.dropout(x)
        print(x.shape)
        x = self.classifier(x)
        print(x.shape)

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



if __name__ == '__main__':
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

    print("deleted_%d size %s model train start!" % (IMAGE_SIZE, MODEL_NAME))
    train_value = []
    valid_value = []
    test_value = []

    # load Dataset
    train_dataset = MyDataset(train_Path,
                              mode='train',
                              transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    print('Train size:', len(train_loader))

    valid_dataset = MyDataset(valid_Path,
                              mode='valid',
                              transform=transforms.Compose([transforms.ToTensor()]))

    valid_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=True)
    print('Valid size:', len(valid_loader))

    test_dataset = MyDataset(test_Path,
                             mode='test',
                             transform=transforms.Compose([transforms.ToTensor()]))

    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)
    print('Test size:', len(test_loader))

    # Training part
    effnet = EfficientNet(param, num_classes=num_classes)
    effnet = effnet.to(device)

    summary(effnet, (3, IMAGE_SIZE, IMAGE_SIZE))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(effnet.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.99))

    start_time = time.time()

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    classes = ['Benign', 'Malware']

    min_val_loss = 0.5
'''
    for epoch in range(1, EPOCH + 1):
        print("Epoch : %d" % epoch)

        train_loss = 0.0
        train_acc = 0.0

        # train
        effnet.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.int64)  # gpu setting

            optimizer.zero_grad()  # grad init
            output = effnet(data)  # forward propagation
            loss = criterion(output, label)  # calculate loss
            loss.backward()  # back propagation
            optimizer.step()  # weight update

            pred = ((output.argmax(dim=1) == label).float().mean())
            train_acc += pred / len(train_loader)
            train_loss += loss / len(train_loader)  # train_loss summary

            # delete for memory issue
            del loss
            del output

        # train_loss /= len(train_loader.dataset)
        print('Train Epoch: {} '
              'Average loss: {:.4f} '
              'Accuracy : {:.4f}%)'.format(epoch,
                                           train_loss,
                                           train_acc))
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        del data
        del label

        # validation
        effnet.eval()
        with torch.no_grad():
            val_loss, val_acc = 0.0, 0.0
            for valid_batch, (data, label) in enumerate(valid_loader):
                data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.int64)

                val_output = effnet(data)
                v_loss = criterion(val_output, label)

                val_pred = ((val_output.argmax(dim=1) == label).float().mean())
                val_acc += val_pred / len(valid_loader)
                val_loss += v_loss / len(valid_loader)

            print('Validation set: '
                  'Average loss: {:.4f}, '
                  'Accuracy: {:.4f}%)'.format(val_loss,
                                              val_acc))
        del data
        del label

        valid_loss_list.append(val_loss)
        valid_acc_list.append(val_acc)
        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            print("min_validation_loss update")
            print("min_val_loss: %.4f" % min_val_loss)
            save_path = './model/deleted_%s_%s_checkpoint_%depoch_%0.4f_%0.4f.pth' % (
            MODEL_NAME, IMAGE_SIZE, epoch, val_acc, val_loss)

            # save check point
            torch.save({
                'epoch': epoch,
                'model_state_dict': effnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss_value,
            }, save_path)

    # open CSV log file
    log_file = open(CSVLog_name, 'w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Epochs', 'Train_Accuracy', 'Train_Loss', 'Valid_Accuracy', 'Valid_Loss'])
    for i in range(0, EPOCH):
        log_writer.writerow([i + 1, train_acc_list[i], train_loss_list[i], valid_acc_list[i], valid_loss_list[i]])


    print(time.time() - start_time)
    print("deleted_%d size %s model train Finished!" % (IMAGE_SIZE, MODEL_NAME))
'''