import argparse
from models import *
from torchinfo import summary
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import torch.utils.data

parser = argparse.ArgumentParser(description='Train CIFAR10 with PyTorch')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


#Data

print('------ Preparing data ------')

batch_size = 64

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR10(
    root='./classify/data', train=True, download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.CIFAR10(
    root='./classify/data', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=0)


#Model

#print('------ Check CNN Model ------')

# net = AlexNet()
# net = VGG16()           # VGG11/13/16/19
# net = GoogLeNet()
# net = ResNet50()        # ResNet18/34/50/101/152
# net = DenseNet121()     # DenseNet121/161/169/201/264

# net = SE_ResNet50()
# net = CBAM_ResNet50()
# net = ECA_ResNet50()

# net = squeezenet1_0()   # squeezenet1_0/1_1
# net = MobileNet()
# net = shufflenet_g8()   # shufflenet_g1/g2/g3/g4/g8
# net = Xception()

best_acc = 0
start_epoch = 0
end_epoch = start_epoch + 10

net = ResNet50()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('cuda is available : ', torch.cuda.is_available())
net = net.to(device)

if args.resume:
    print('------ Loading checkpoint ------')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    end_epoch += start_epoch

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)


#Model Summary

def model_summary():
    print('------ Model Summary ------')
    y = net(torch.randn(1, 3, 32, 32).to(device))
    print(y.size())
    summary(net, (1, 3, 32, 32), depth=5)


#Training

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        
        if batch_idx % 50 == 0:
            print('\tLoss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    train_loss = train_loss / len(train_loader)
    train_acc = 100. * correct / total

    print('\n', time.asctime(time.localtime(time.time())))
    print(' Epoch: %d | Train_loss: %.3f | Train_acc: %.3f%% \n' % (epoch, train_loss, train_acc))

    return train_loss, train_acc


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                print('\tLoss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total

        print('\n', time.asctime(time.localtime(time.time())))
        print(' Epoch: %d | Test_loss: %.3f | Test_acc: %.3f%% \n' % (epoch, test_loss, test_acc))

    if test_acc > best_acc:
        print('------ Saving model------')
        state = {
            'net': net.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/model_%d_%.3f.pth' % (epoch, best_acc))
        best_acc = test_acc

    return test_loss, test_acc


def save_csv(epoch, save_train_loss, save_train_acc, save_test_loss, save_test_acc):
    time = '%s' % datetime.now()
    step = 'Step[%d]' % epoch
    train_loss = '%f' % save_train_loss
    train_acc = '%g' % save_train_acc
    test_loss = '%f' % save_test_loss
    test_acc = '%g' % save_test_acc

    print('------ Saving csv ------')
    list = [time, step, train_loss, train_acc, test_loss, test_acc]

    data = pd.DataFrame([list])
    data.to_csv('./train_acc.csv', mode='a', header=False, index=False)


def draw_acc():
    filename = r'./train_acc.csv'

    train_data = pd.read_csv(filename)
    print(train_data.head())

    length = len(train_data['step'])
    Epoch = list(range(1, length + 1))

    train_loss = train_data['train loss']
    train_accuracy = train_data['train accuracy']
    test_loss = train_data['test loss']
    test_accuracy = train_data['test accuracy']

    plt.plot(Epoch, train_loss, 'g-.', label='train loss')
    plt.plot(Epoch, train_accuracy, 'r-', label='train accuracy')
    plt.plot(Epoch, test_loss, 'b-.', label='test loss')
    plt.plot(Epoch, test_accuracy, 'm-', label='test accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Loss & Accuracy')
    plt.yticks([j for j in range(0, 101, 10)])
    plt.title('Epoch -- Loss & Accuracy')

    plt.legend(loc='center right', fontsize=8, frameon=False)
    plt.show()


if __name__ == '__main__':

    model_summary()
    if not os.path.exists('../GCN/train_acc.csv'):
        df = pd.DataFrame(columns=['time', 'step', 'train loss', 'train accuracy', 'test loss', 'test accuracy'])
        df.to_csv('./train_acc.csv', index=False)
        print('make csv successful !')
    else:
        print('csv is exist !')

    for epoch in range(start_epoch, end_epoch):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        scheduler.step()

        save_csv(epoch, train_loss, train_acc, test_loss, test_acc)

    draw_acc()
