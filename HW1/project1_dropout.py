import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torch import optim
import matplotlib.pyplot as plt
from os import makedirs
from os.path import exists

def initialization():
    class GPUDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, device=None):
            if device is None:
                device = 'cuda'
            self.device = torch.device(device) if isinstance(device, str) else device
            # 将 numpy 的图像数据转换为 tensor，并把通道维移动到前面 (N, C, H, W)，同时归一化到 [0,1]
            data = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float() / 255.0

            # 使用 CIFAR-10 常用的均值与方差进行归一化（提高训练稳定性）
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
            data = (data - mean) / std

            # 将数据和标签移动到指定设备（通常为 GPU），以便在训练时直接使用
            self.data = data.to(self.device)
            self.targets = torch.tensor(dataset.targets, dtype=torch.long).to(self.device)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    # 选择设备：若有 GPU 可用则使用 GPU，否则使用 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_save_path = './results_dropout/models'
    plot_save_path = './results_dropout/plots'
    data_save_path = './data'

    if not exists(model_save_path):
        makedirs(model_save_path)
    if not exists(plot_save_path):
        makedirs(plot_save_path)
    if not exists(data_save_path):
        makedirs(data_save_path)

    # 用于将 tensor 转回 PIL Image 以便展示
    show = ToPILImage()
    # 保留的 transform（演示用途）：将图像转为 tensor 并按给定均值/方差归一化
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    batch_size = 64

    # 下载 CIFAR-10 数据集（若本地已有则不重复下载）
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    # 将数据封装为 GPUDataset，直接把数据放到 GPU（device）上，加快后续训练
    train_gpu = GPUDataset(trainset, device='cuda')
    test_gpu = GPUDataset(testset, device='cuda')

    # DataLoader 负责按批次读取数据；num_workers=0 在 Windows 上更兼容
    trainloader = DataLoader(train_gpu, batch_size=64, shuffle=True, num_workers=0)
    testloader = DataLoader(test_gpu, batch_size=64, shuffle=False, num_workers=0)
    print("DataLoader ready. Dataset downloaded.")

    l2_lambda = 0.0001

    class Net(nn.Module): #with Dropout
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            # Dropout：在训练时随机丢弃部分神经元，减轻过拟合
            self.dropout = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(x.size()[0], -1)
            x = F.relu(self.fc1(x))
            # 在全连接层间使用 Dropout（训练模式生效，eval 模式自动禁用）
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net().to(device)

    print("model initialized.")

    # 交叉熵损失用于多分类任务
    criterion = nn.CrossEntropyLoss()
    # SGD 优化器：学习率较小，带动量并使用 L2 权重衰减（weight_decay）作为正则化
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=l2_lambda)
    num_epochs = 100

    print("Initialization complete.")
    print(f"Model is on device: {next(net.parameters()).device}")

    return (model_save_path, plot_save_path, num_epochs, criterion,
            optimizer, trainloader, testloader, net, show, transform,
            batch_size, trainset, testset, device, l2_lambda)


def draw_loss_and_accuracy_curve(loss, steps, train_acc, test_acc, epochs, path):
    fig, ax1 = plt.subplots()
    ax1.plot(steps, loss, 'b-', label='Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(steps, train_acc, 'r-', label='Train Accuracy')
    ax2.plot(steps, test_acc, 'g-', label='Test Accuracy')
    ax2.set_ylabel('Accuracy', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f'Training Loss and Accuracy Curve over {epochs} Epochs')
    fig.tight_layout()
    plt.savefig(f"{path}/loss_and_accuracy_curve.png")
    plt.close()
    # 保存并关闭图像文件，后续可以查看 results_dropout/plots 中的图片

def predict(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # 使用自动混合精度进行前向推断（若 GPU 支持），可提升速度并节省显存
            with torch.amp.autocast("cuda"):
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    acc = 100 * correct / total
    return acc.item()

def train(train_loader, test_loader, model, epochs, crit, opti, save_path, device):
    from torch import amp

    # 使用混合精度训练时的 GradScaler，可以对 loss 放缩以避免半精度下梯度下溢
    scaler = amp.GradScaler("cuda")

    print("Start Training...")
    epochs_list = []
    losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        epoch_loss = 0.0  # 初始化每轮损失
        train_correct = 0
        train_total = 0

        model.train()
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

                # 统计训练集预测正确数（在同一前向输出上）
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # 3. 反向传播，更新参数（缩放梯度以处理半精度）
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            scaler.step(opti)  # 更新优化器
            scaler.update()  # 更新 scaler 以处理梯度裁剪

            # 累积损失（注意：loss.item() 是 float32 值）
            epoch_loss += loss.item()

        # 计算每轮平均损失与训练准确率
        avg_loss = epoch_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0

        # 保存模型参数（state_dict），便于之后加载或继续训练
        torch.save(model.state_dict(), f"{save_path}/epoch_{epoch + 1}_model.pth")

        # 调用predict并获取测试集准确率
        test_acc = predict(test_loader, model, device)

        # 输出轮数、平均损失和准确率
        print(f'轮数: {epoch + 1}, 平均损失: {avg_loss:.3f}, 训练准确率: {train_acc:.2f}%, 测试准确率: {test_acc:.2f}%')

        # 收集数据到数组
        epochs_list.append(epoch + 1)
        losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    print('Finished Training')
    return epochs_list, losses, train_accuracies, test_accuracies

if __name__ == "__main__":

    (model_path, plot_path, num_of_epochs, Criterion,
     Optimizer, trainLoader, testLoader, network, Show, Transform,
     batchSize, trainSet, testSet, DEVICE, l2) = initialization()

    if torch.cuda.is_available():
        test_tensor = torch.randn(10000, 10000).to(DEVICE)
        result = torch.matmul(test_tensor, test_tensor)
        print("GPU test completed")

    epoch_list, loss_list, train_accuracy_list, test_accuracy_list = train(trainLoader, testLoader, network,
                                                                           num_of_epochs,Criterion, Optimizer,
                                                                           model_path, DEVICE)

    draw_loss_and_accuracy_curve(loss_list, epoch_list, train_accuracy_list, test_accuracy_list, num_of_epochs,
                                 plot_path)
