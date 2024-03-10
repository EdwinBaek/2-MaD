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
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Hyper-Parameters
IMAGE_SIZE = 448
MODEL_NAME = 'ResNeXt50'
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


# ResNeXt101 modeling
def conv_start():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=4),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )

def bottleneck_block(in_dim, mid_dim, out_dim, groups=32, down=False):
    layers = []
    width = mid_dim // 64 * 32 * 4
    k = 2 if down else 1
    layers.append(nn.Conv2d(in_dim, width, kernel_size=1, stride=k, padding=0))

    layers.extend([
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
        nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, groups=groups),
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
        nn.Conv2d(width, out_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_dim),
    ])
    return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, down: bool = False, starting: bool = False) -> None:
        super(Bottleneck, self).__init__()
        if starting:
            down = False
        self.block = bottleneck_block(in_dim, mid_dim, out_dim, down=down)
        self.relu = nn.ReLU(inplace=True)
        k = 2 if down else 1

        self.changedim = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=k, padding=0),
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


class ResNeXt(nn.Module):
    def __init__(self, repeats:list = [3, 4, 6, 3], num_classes=1000):
        super(ResNeXt, self).__init__()
        self.num_classes = num_classes
        self.conv1 = conv_start()

        base_dim = 64
        self.conv2 = make_layer(base_dim, base_dim, base_dim*4, repeats[0], starting=True)
        self.conv3 = make_layer(base_dim*4, base_dim*2, base_dim*8, repeats[1])
        self.conv4 = make_layer(base_dim*8, base_dim*4, base_dim*16, repeats[2])
        self.conv5 = make_layer(base_dim*16, base_dim*8, base_dim*32, repeats[3])

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifer = nn.Linear(2048 * 8 * 8, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        return x




if __name__ == '__main__':
    print("deleted_%d size %s model train start!" % (IMAGE_SIZE, MODEL_NAME))
    train_value = []
    valid_value = []
    test_value = []

    # load Dataset
    train_dataset = MyDataset(train_Path,
                              mode='train',
                              transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    #print('Train size:', len(train_loader))

    valid_dataset = MyDataset(valid_Path,
                              mode='valid',
                              transform=transforms.Compose([transforms.ToTensor()]))

    valid_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=True)
    #print('Valid size:', len(valid_loader))

    test_dataset = MyDataset(test_Path,
                             mode='test',
                             transform=transforms.Compose([transforms.ToTensor()]))

    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)
    #print('Test size:', len(test_loader))

    # Training part
    resnext = ResNeXt(num_classes=num_classes)
    resnext = resnext.to(device)

    summary(resnext, (3, IMAGE_SIZE, IMAGE_SIZE))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(resnext.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.99))

    start_time = time.time()

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    min_val_loss = 0.5

    for epoch in range(1, EPOCH + 1):
        print("Epoch : %d" % epoch)

        train_loss = 0.0
        train_acc = 0.0

        # train
        resnext.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.int64)  # gpu setting

            optimizer.zero_grad()  # grad init
            output = resnext(data)  # forward propagation
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
        resnext.eval()
        with torch.no_grad():
            val_loss, val_acc = 0.0, 0.0
            for valid_batch, (data, label) in enumerate(valid_loader):
                data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.int64)

                val_output = resnext(data)
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
                'model_state_dict': resnext.state_dict(),
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