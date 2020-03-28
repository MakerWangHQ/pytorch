#%%

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import random
import torch.optim as optim
import time

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
print(net)


def imshow(img):
    img = img / 2 + 0.5     # 把数据退回(0,1)区间
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)


train_dataset = torchvision.datasets.CIFAR10(root='data/',train=True,transform=transforms,download=True)
test_dataset = torchvision.datasets.CIFAR10(root='data/',train=False,transform=transforms,download=True)
batch_size = 4

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2) 
dataiter = iter(trainloader)
images, labels = dataiter.next()
images, labels = dataiter.next()
print(images.size())

# 展示图片
imshow(torchvision.utils.make_grid(images))
# 展示分类
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

net = Net()
criterion2 = nn.CrossEntropyLoss()
optim = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs,labels = data
        # print('inputs.size', inputs.size())
        
        optim.zero_grad()
        
        outputs = net(inputs)
        # print('outputs.size()', outputs.size())
        loss = criterion2(outputs, labels)
        loss.backward()
        optim.step()
        
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i+1, running_loss / 2000))
            running_loss = 0.0

stop_time = time.time()
print('训练用时', stop_time - start_time, '秒')
print('Finished Training')

#%%

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth:', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

#%%

outputs = net(images)
_, predicted = torch.max(outputs,1)
print('Predicted:', ''.join('%5s' % classes[predicted[j]] for j in range(4)))

#%%

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images:\n %d %%' %(100 * correct /total))


#%%

PATH = '.\cifar_net.pth'
torch.save(net.state_dict(), PATH)
print(net.state_dict())
