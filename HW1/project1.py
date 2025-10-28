import os
from HW1.train import train, device, trainloader, testloader, batch_size, predict

save_path = './HW1/results'
os.makedirs(save_path, exist_ok=True)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()

        # 卷积层 '3'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层，y = Wx + b
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积 -> 激活 -> 池化 (relu激活函数不改变输入的形状)
        # [batch size, 3, 32, 32] -- conv1 --> [batch size, 6, 28, 28] -- maxpool --> [batch size, 6, 14, 14]
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # [batch size, 6, 14, 14] -- conv2 --> [batch size, 16, 10, 10] --> maxpool --> [batch size, 16, 5, 5]
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 把 16 * 5 * 5 的特征图展平，变为 [batch size, 16 * 5 * 5]，以送入全连接层
        x = x.view(x.size()[0], -1)
        # [batch size, 16 * 5 * 5] -- fc1 --> [batch size, 120]
        x = F.relu(self.fc1(x))
        # [batch size, 120] -- fc2 --> [batch size, 84]
        x = F.relu(self.fc2(x))
        # [batch size, 84] -- fc3 --> [batch size, 10]
        x = self.fc3(x)
        return x

model = Net()
model.to(device)

from torch import optim
criterion = nn.CrossEntropyLoss() .to(device)# 交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # 使用SGD（随机梯度下降）优化
num_epochs = int(input("How many epoch do you want to train: "))  # 训练 x 个 epoch

loss = train(trainloader, model, num_epochs, criterion, optimizer, save_path)
accuracy = predict(testloader, model)

import matplotlib.pyplot as plt

def draw(values, plot_dir='./HW1/results/plots', filename=None, dpi=150):
    """
    将折线图保存到指定文件夹。默认目录为 `./results/plots`。
    """
    os.makedirs(plot_dir, exist_ok=True)
    if filename is None:
        filename = f"loss_batchsize{batch_size}_epochs{num_epochs}_acc{accuracy}.png"
    path = os.path.join(plot_dir, filename)

    plt.figure()
    plt.plot(values)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to `{path}`")

draw(loss)