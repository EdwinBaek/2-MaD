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
MODEL_NAME = 'VGG16'
CSVLog_name = './logs/%s_%s_logs.csv' % (MODEL_NAME, IMAGE_SIZE)
train_Path = './data/npy/train%s/' % IMAGE_SIZE
valid_Path = './data/npy/valid%s/' % IMAGE_SIZE
test_Path = './data/npy/test%s/' % IMAGE_SIZE
num_classes = 2
BATCH = 1
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
        train_ID, train_Label = getData('./data/labels/train_ID_Labels.csv')
        valid_ID, valid_Label = getData('./data/labels/valid_ID_Labels.csv')
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


# VGG modeling
class VGG16(nn.Module):
    def __init__(self, num_classes: int = 1000, init_weights: bool = True):
        super(VGG16, self).__init__()
        self.convnet = nn.Sequential(
            # Input Channel (RGB: 3)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fclayer = nn.Sequential(
            nn.Linear(512 * 14 * 14, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
            #nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = self.convnet(x)
        x = torch.flatten(x, 1)
        x = self.fclayer(x)
        return x


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    train_acc = 0
    print_train_loss = 0
    print_train_acc = 0
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)
        label = label.type(torch.Tensor).cuda()
        pred = torch.max(output, 1)[1]

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print_train_loss += loss.item()
        train_acc += torch.sum(pred == label.data)
        print_train_acc += torch.sum(pred == label.data)

        if (batch_idx + 1) % 200 == 0:
            train_loss /= 200
            train_acc = train_acc.double() / 200
            print('Train Epoch: %d [%d/%d (%.4f%%)\tLoss: %.4f\tAccuracy : %.4f]' % (
                epoch, (batch_idx + 1) * len(image), len(train_loader.dataset),
                100. * (batch_idx + 1) * len(image) / len(train_loader.dataset), train_loss, train_acc))
            train_loss = 0

        if (batch_idx + 1) % len(train_loader) == 0:
            print_train_loss /= len(train_loader)
            print_train_acc = print_train_acc.double() / len(train_loader)
            print('Train Epoch: %d [%d/%d (%.4f%%)\tLoss: %.4f\tAccuracy : %.4f]' % (
                epoch, (batch_idx + 1) * len(image), len(train_loader.dataset),
                100. * (batch_idx + 1) * len(image) / len(train_loader.dataset), print_train_loss, print_train_acc))

            train_value.append([print_train_loss, (float)(print_train_acc)])
            print_train_loss = 0
        global train_acc_value, train_loss_value
        train_acc_value, train_loss_value = train_acc, train_loss


def valid(model, device, valid_loader, criterion, epoch):
    model.eval()
    total_true = 0
    total_loss = 0

    with torch.no_grad():
        for image, label in valid_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            label = label.type(torch.Tensor).cuda()
            loss = criterion(output, label)
            pred = torch.max(output, 1)[1]
            total_true += (pred.view(label.size()).data == label.data).sum().item()
            total_loss += loss.item()

    accuracy = total_true / len(valid_loader.dataset)
    loss = total_loss / len(valid_loader.dataset)

    print('\nValidation Epoch: %d ====> Accuracy: [%d/%d (%.4f%%)]\tAverage loss: %.4f\n' % (epoch, total_true, len(valid_loader.dataset), 100. * accuracy, loss))
    valid_value.append([loss, accuracy])
    global valid_acc_value, valid_loss_value
    valid_acc_value, valid_loss_value = accuracy, loss


'''
def test(model, device, test_loader):
    model.eval()
    test_pred = torch.LongTensor()
    total_loss = 0

    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            label = label.type(torch.LongTensor).cuda()

            loss = criterion(output, label)
            pred = output.cpu().data.max(1, keepdim=True)[1]
            test_pred = torch.cat((test_pred, pred), dim=0)
            total_loss += loss.item()

    out_df = pd.DataFrame(np.c_[np.arange(1, len(test_dataset)+1)[:,None], test_pred.numpy()], columns=['ImageID', 'Label'])
    print(out_df.head())
    out_df.to_csv('./data/submission.csv', index=False)

    test_Label = getData('./labels/test_Label.csv')
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

    print('\nTest loss >>>>> (%.4f%%)'% (loss))
    print('\nTest accuracy >>>>> (%.4f%%)'% (accuracy_score(y_true, y_pred)))
    test_value.append([loss, accuracy_score(y_true, y_pred), tp, fn, fp, fn])
'''


if __name__ == '__main__':
    print("%d size %s model train start!" % (IMAGE_SIZE, MODEL_NAME))
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
    vgg16 = VGG16(num_classes=num_classes)
    vgg16 = vgg16.to(device)

    summary(vgg16, (3, IMAGE_SIZE, IMAGE_SIZE))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(vgg16.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.99))

    start_time = time.time()

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    classes = ['Benign', 'Malware']

    min_val_loss = 0.5

    for epoch in range(1, EPOCH + 1):
        print("Epoch : %d" % epoch)

        train_loss = 0.0
        train_acc = 0.0

        # train
        vgg16.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.int64)    # gpu setting

            optimizer.zero_grad()               # grad init
            output = vgg16(data)                # forward propagation
            loss = criterion(output, label)     # calculate loss
            loss.backward()                     # back propagation
            optimizer.step()                    # weight update

            pred = ((output.argmax(dim=1) == label).float().mean())
            train_acc += pred / len(train_loader)
            train_loss += loss / len(train_loader)         # train_loss summary

            # delete for memory issue
            del loss
            del output

        #train_loss /= len(train_loader.dataset)
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
        vgg16.eval()
        with torch.no_grad():
            val_loss, val_acc = 0.0, 0.0
            for valid_batch, (data, label) in enumerate(valid_loader):
                data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.int64)

                val_output = vgg16(data)
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
            save_path = './model/%s_%s_checkpoint_%depoch_%0.4f_%0.4f.pth' % (MODEL_NAME, IMAGE_SIZE, epoch, val_acc, val_loss)

            # save check point
            torch.save({
                'epoch': epoch,
                'model_state_dict': vgg16.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss_value,
            }, save_path)


    # open CSV log file
    log_file = open(CSVLog_name, 'w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Epochs', 'Train_Accuracy', 'Train_Loss', 'Valid_Accuracy', 'Valid_Loss'])
    for i in range(0, EPOCH):
        log_writer.writerow([i + 1, train_acc_list[i], train_loss_list[i], valid_acc_list[i], valid_loss_list[i]])


    '''
    for epoch in range(100):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # print(inputs.shape)
            outputs = vgg19(inputs)

            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:    # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    # Validation
    vgg19.eval()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = vgg19(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1  
    

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    '''

    print(time.time() - start_time)
    print("%d size %s model train Finished!" % (IMAGE_SIZE, MODEL_NAME))