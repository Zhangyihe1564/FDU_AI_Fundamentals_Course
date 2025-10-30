import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch import optim
import matplotlib.pyplot as plt
from os import makedirs
from os.path import exists

def initialization():

    class GPUDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transforms=None, device=None):
            if device is None:
                device = 'cuda'  # 默认str
            # 自动转换：如果device是str，转为torch.device；如果是torch.device，保持
            device = torch.device(device) if isinstance(device, str) else device

            # 加载数据到GPU：从numpy转为tensor，调整通道顺序，归一化到[0,1]，移到device
            self.data = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float().to(device) / 255.0
            self.targets = torch.tensor(dataset.targets).to(device)
            self.transform = transforms

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            img, label = self.data[idx], self.targets[idx]
            if self.transform:
                img = self.transform(img)  # Normalize等在GPU上运行（torchvision支持）
            return img, label

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_save_path = './results_normal/models'
    plot_save_path = './results_normal/plots'
    data_save_path = './data'

    if not exists(model_save_path):
        makedirs(model_save_path)
    if not exists(plot_save_path):
        makedirs(plot_save_path)
    if not exists(data_save_path):
        makedirs(data_save_path)

    show = ToPILImage()
    # 设定对图片的归一化处理方式，并且下载数据集
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    norm_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root=data_save_path, train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_save_path, train=False,
                                           download=True, transform=transform)

    trainset_gpu = GPUDataset(trainset, norm_transform, device=device)
    testset_gpu = GPUDataset(testset, norm_transform, device=device)

    trainloader = torch.utils.data.DataLoader(trainset_gpu, batch_size=batch_size, shuffle=True,
                                          num_workers=0)  # num_workers=0，因为已在GPU
    testloader = torch.utils.data.DataLoader(testset_gpu, batch_size=batch_size, shuffle=False, num_workers=0)
    print("DataLoader ready. Dataset downloaded.")


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

    net = Net().to(device)

    print("model initialized.")

    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # 使用SGD（随机梯度下降）优化
    num_epochs = 20

    print("Initialization complete.")
    print(f"Model is on device: {next(net.parameters()).device}")

    return (model_save_path, plot_save_path, num_epochs, criterion,
            optimizer, trainloader, testloader, net, show, transform,
            batch_size, trainset, testset, device)


def draw_loss_and_accuracy_curve(loss, steps, acc, epochs, path):
    fig, ax1 = plt.subplots()
    ax1.plot(steps, loss, 'b-', label='Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(steps, acc, 'r-', label='Accuracy')
    ax2.set_ylabel('Accuracy', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title(f'Training Loss and Accuracy Curve over {epochs} Epochs')
    fig.tight_layout()
    plt.savefig(f"{path}/loss_and_accuracy_curve.png")
    plt.close()


def predict(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda"):  # AMP加速测试
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    acc = 100 * correct / total
    return acc.item()


def train(train_loader, test_loader, model, epochs, crit, opti, save_path, device):
    from torch import amp  # 使用新的 torch.amp 模块

    scaler = amp.GradScaler("cuda")  # 初始化 GradScaler，指定 "cuda" 以避免弃用警告

    print("Start Training...")
    epochs_list = []
    losses = []
    accuracies = []
    for epoch in range(epochs):
        epoch_loss = 0.0  # 初始化每轮损失
        for i, data in enumerate(train_loader, 0):
            # 1. 取出数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # 梯度清零
            opti.zero_grad()

            # 2. 前向计算和反向传播（使用混合精度）
            with amp.autocast("cuda"):  # 启用自动混合精度，指定 "cuda"
                outputs = model(inputs)  # 送入网络（正向传播）
                loss = crit(outputs, labels)  # 计算损失函数

            # 3. 反向传播，更新参数（缩放梯度以处理半精度）
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            scaler.step(opti)  # 更新优化器
            scaler.update()  # 更新 scaler 以处理梯度裁剪

            # 累积损失（注意：loss.item() 是 float32 值）
            epoch_loss += loss.item()

        # 计算每轮平均损失
        avg_loss = epoch_loss / len(train_loader)

        # 保存模型
        torch.save(model.state_dict(), f"{save_path}/epoch_{epoch + 1}_model.pth")

        # 调用predict并获取准确率
        acc = predict(test_loader, model, device)

        # 输出轮数、平均损失和准确率
        print(f'轮数: {epoch + 1}, 平均损失: {avg_loss:.3f}, 准确率: {acc:.2f}%')

        # 收集数据到数组
        epochs_list.append(epoch + 1)
        losses.append(avg_loss)
        accuracies.append(acc)

    print('Finished Training')
    return epochs_list, losses, accuracies
# 在主代码中调用train并绘图
if __name__ == "__main__":

    (model_path, plot_path, num_of_epochs, Criterion,
     Optimizer, trainLoader, testLoader, network, Show, Transform,
     batchSize, trainSet, testSet, DEVICE) = initialization()

    print(torch.cuda.is_available())  # 应为 True
    print(torch.cuda.get_device_name(0))  # 显示你的 GPU 名，如 'NVIDIA GeForce RTX 4060 Laptop GPU'
    print(next(network.parameters()).device)  # 应为 'cuda:0'

    if torch.cuda.is_available():
        test_tensor = torch.randn(10000, 10000).to(DEVICE)
        result = torch.matmul(test_tensor, test_tensor)
        print("GPU test completed")

    epoch_list, loss_list, accuracy_list = train(trainLoader, testLoader, network, num_of_epochs,
                                                 Criterion, Optimizer, model_path, DEVICE)

    draw_loss_and_accuracy_curve(loss_list, epoch_list, accuracy_list, num_of_epochs, plot_path)
