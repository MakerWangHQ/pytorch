import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt

import datetime
import argparse


# 样本读取线程数
WORKERS = 4

#网络参数保存文件名
PARAS_FN = 'cifar_reset_params.pth'

#数据存放的位置
ROOT = './data'

# 目标函数
loss_func = nn.CrossEntropyLoss()

# 最优结果
best_acc = 0

# 记录准确率，显示曲线
global_train_acc = []
global_test_acc = []

'''
残差块
in_channels, out_channels:残差块的输入，输出通道数
对第一层，in out channel 都是64，其他层则不同
对每一层，如果in out channel不同，stride是1，其他层则为2
'''

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # 残差块的第一个卷积
        # 通道数变换in->out, 每一层（除第一层外）的第一个block
        # 图片尺寸变换：stride=2时，w-3+2/2+1 = w/2, w/2 * w/2
        # stride=1时尺寸不变，w-3+2 / 1 = w
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 残差块的第二个卷积
        # 通道数、图片尺寸均不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差块的shortcut
        # 如果残差块的输入输出通道数不同，则需要变换通道数及图片尺寸，以和residual部分相加
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            #通道数相同，无需做变换，在forward中identity=x
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

'''
定义网络结构
'''
class ResNet34(nn.Module):
    def __init__(self, block):
        super(ResNet34, self).__init__()

        # 初始卷积层和池化层
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, 1, 1)
        )


        self.layer1 = self.make_layer(block, 64, 64, 3, 1)

        self.layer2 = self.make_layer(block, 64, 128, 4, 2)
        self.layer3 = self.make_layer(block, 128, 256, 6, 2)
        self.layer4 = self.make_layer(block, 256, 512, 3, 2)

        self.avg_pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512, 10)

    def make_layer(self, block, in_channels, out_channels, block_num, stride):
        layers = []

        layers.append(block(in_channels, out_channels, stride))

        for i in range(block_num - 1):
            layers.append(block(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


'''
训练并测试网络
net: 网络模型
train_data_load: 训练数据集
optimizer: 优化器
epoch: 第几次训练迭代
log_interval: 训练过程中损失函数值和准确率的打印频率
'''

def net_train(net, train_data_load, optimizer, epoch, log_interval):
    net.train()

    begin = datetime.datetime.now()

    total = len(train_data_load.dataset)

    train_loss = 0

    ok = 0

    for i, (img, label) in enumerate(train_data_load, 0):
        img, label = img.cuda(), label.cuda()

        optimizer.zero_grad()

        outs = net(img)
        loss = loss_func(outs, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outs.data, 1)

        ok += (predicted == label).sum()

        if(i + 1) % log_interval == 0:

            trained_total = (i+1) * len(label)

            acc = 100. * ok / trained_total

            global_train_acc.append(acc)

    end = datetime.datetime.now()
    print('one epoch spend: ', end - begin)


'''
用测试集检查准确率
'''
def net_test(net, test_data_load, epoch):
    net.eval()

    ok = 0

    for i, (img, label) in enumerate(test_data_load):
        img, label = img.cuda(), label.cuda()

        outs = net(img)
        _, pre = torch.max(outs.data, 1)
        ok += (pre == label).sum()

    acc = ok.item() * 100. / (len(test_data_load.dataset))
    print('EPOCH:{}, ACC:{}\n'.format(epoch, acc))

    global_test_acc.append(acc)

    global best_acc
    if acc > best_acc:
        best_acc = acc

'''
显示数据集中一个图片
'''
def img_show(dataset, index):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    show = ToPILImage()

    data, label = dataset[index]

    print('img is a ', classes[label])
    show((data + 1) / 2).resize((100, 100)).show()

'''
显示训练准确率、测试准确率变化曲线
'''
def show_acc_curv(ratio):
    train_x = list(range(len(global_train_acc)))
    train_y = global_train_acc

    test_x = train_x[ratio-1::ratio]
    test_y = global_test_acc

    plt.title('CIFAR10 RESNET34 ACC')

    plt.plot(train_x, train_y, color='green', label='training accurary')
    plt.plot(test_x, test_y, color='red', label='testing accurary')

    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('accs')

    plt.show()

def main():
    # 训练超参数设置，可通过命令行设置
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 LeNet Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default:64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default:1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default:20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default : 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default:0.9)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before loging training status (default:100)')
    parser.add_argument('--no-train', action='store_true', default=False, help='If train the Model')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    # 图像数值转换，ToSensor源码注释
    """Convert a ''PIL Image'' or numpy.ndarray'' to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0,255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    """
    # 归一化把[0.0,1.0]变换为[-1,1]， ([0,1] - 0.5) / 0.5 = [-1, 1]
    transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # 定义数据集
    train_data = torchvision.datasets.CIFAR10(root=ROOT, train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root=ROOT, train=False, download=True, transform=transform)

    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=WORKERS)
    test_load = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True,
                                            num_workers=WORKERS)

    net = ResNet34(ResBlock).cuda()
    print(net)

    #并行计算提高运行速度
    net = nn.DataParallel(net)
    cudnn.benchmark = True

    # 如果不训练，直接加载保存的网络参数进行测试集验证
    if args.no_train:
        net.load_state_dict(torch.load(PARAS_FN))
        net_test(net, test_load, 0)
        return

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    start_time = datetime.datetime.now()

    for epoch in range(1, args.epochs + 1):
        net_train(net, train_load, optimizer, epoch, args.log_interval)

        # 每个epoch结束后用测试集检查识别准确度
        net_test(net, test_load, epoch)

    end_time = datetime.datetime.now()

    global best_acc
    print(
        'CIFAR10 pytorch ResNet34 Train: EPOCH:{}, BATCH_SZ:{}, LR:{}, ACC:{}'.format(args.epochs, args.batch_size, args.lr, best_acc))
    print('train spend time:', end_time - start_time)

    # 每训练一个迭代记录的训练准确率个数
    ratio = len(train_data) / args.batch_size / args.log_interval
    ratio = int(ratio)

    #显示曲线
    show_acc_curv(ratio)


    if args.save_model:
        torch.save(net.state_dict(), PARAS_FN)


if __name__ == '__main__':
    main()
