#%%
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import os
from HW1.train_and_predict import train, predict

save_path = './HW1/results_regularization'
os.makedirs(save_path, exist_ok=True)

show = ToPILImage()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%%
# 设定对图片的归一化处理方式，并且下载数据集
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%
import torch.nn as nn
import torch.nn.functional as F


# python
class Dropout_Net(nn.Module):
    def __init__(self):
        super(Dropout_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.dropout = nn.Dropout(p=0.5)  # 在第一个线性层和第二个线性层之间加入 Dropout
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)            # Dropout 在训练时生效，评估时自动禁用
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

dropout_net = Dropout_Net()
dropout_net.to(device)

#%%
from torch import optim

criterion = nn.CrossEntropyLoss().to(device)  # 交叉熵损失函数
decay_params = []
no_decay_params = []
for name, param in dropout_net.named_parameters():
    if not param.requires_grad:
        continue
    if name.endswith(".bias") or "bn" in name.lower() or "batchnorm" in name.lower():
        no_decay_params.append(param)
    else:
        decay_params.append(param)

param_groups = [
    {"params": decay_params, "weight_decay": 1e-4},
    {"params": no_decay_params, "weight_decay": 0.0},
]
optimizer = optim.SGD(param_groups, lr=0.001, momentum=0.9)  # 使用L2正则化的随机梯度下降优化器
num_epochs = int(input("How many epoch do you want to train: "))  # 训练 x 个 epoch

#%%
loss, steps = train(trainloader, dropout_net,
                    num_epochs, criterion, optimizer, save_path)
#%%
accuracy = predict(testloader, dropout_net)
#%%
import matplotlib.pyplot as plt


def draw(values, x_values=None, plot_dir='./HW1/results_regularization/plots', filename=None, dpi=150):
    """
    如果提供 x_values，则横轴使用这些全局 step（batch 编号），否则使用索引。
    """
    os.makedirs(plot_dir, exist_ok=True)
    if filename is None:
        filename = f"loss_batchsize{batch_size}_epochs{num_epochs}_acc{accuracy}_dropout_l2r.png"
    path = os.path.join(plot_dir, filename)

    plt.figure()
    if x_values is None:
        plt.plot(values, marker='o')
        plt.xlabel('record index')
    else:
        plt.plot(x_values, values, marker='o')
        plt.xlabel('global step (batch number)')
    plt.ylabel('loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to `{path}`")


draw(loss, steps)
