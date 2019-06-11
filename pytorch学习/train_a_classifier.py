'''
训练图像分类器：
	1.使用torchvision加载并且标准化训练集和测试集
	2.定义卷积神经网络
	3.定义损失函数
	4.使用训练数据训练网络
	5.使用测试数据测试网络
'''

# step 1
# The output of torchvision datasets are PILImage images of range [0, 1].We transform them to Tensors of normalized range [-1, 1].

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 2)
testset = torchvision.datasets.CIFAR10(root='./data', train = False, download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 显示图片
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
	img =  img/2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	
dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' classes[labels[j]] for j in range(4)))

# step 2
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 *5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
		
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 *5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

net = Net()

# step 3
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# step 4
for epoch in range(2):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# 获取输入
		inputs, labels = data
		# 将梯度置0
		optimizer.zero_grad()
		# forward+backward+optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		# 输出统计数据
		running_loss += loss.item()
		if i % 2000 == 1999: #每2000个mini-batches输出一次
			print('[%d, %5d] loss: %.3f' %(epoch+1, i+1, running_loss/2000))
			
	print('Finish Training')

# step 5
dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('GroundTruth:',' '.join('%5s' % classes[labels[j]] for j in range(4)))

output = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
	for data in testloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
		
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# train on GPU
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

net.to(device)
inputs, labels =  data[0].to(device), data[1].to(device)