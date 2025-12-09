import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
from os import makedirs
from os.path import exists
import random
import numpy as np

def set_seed(seed=42):
    # 设置随机种子，避免实验结果波动（尽量可复现）
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 强制确定性设置（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialization():
    # 初始化设备、数据变换、数据集、模型、损失、优化器和学习率调度器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # CIFAR-10 标准均值与方差，用于标准化
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # 训练集的数据增强与预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),        # 随机裁剪并填充
        transforms.RandomHorizontalFlip(),           # 随机翻转
        transforms.ToTensor(),
        transforms.Normalize(mean, std),             # 标准化
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))  # 随机擦除
    ])

    # 测试集只做标准化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 加载 CIFAR-10 数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 以下为 DenseNet 构件定义：Bottleneck、DenseBlock、Transition、以及完整网络

    class Bottleneck(nn.Module):
        def __init__(self, in_channels, growth_rate):
            super().__init__()
            # 两个 BN-ReLU-Conv 的组合，第一层 1x1 减少维度，第二层 3x3 生成增长特征
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(4 * growth_rate)
            self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

        def forward(self, x):
            out = F.relu(self.bn1(x))
            out = F.relu(self.bn2(self.conv1(out)))
            out = self.conv2(out)
            # 残差式连接：将输入与新特征按通道拼接（DenseNet 的关键）
            return torch.cat([x, out], 1)

    class DenseBlock(nn.Module):
        def __init__(self, n_layers, in_channels, growth_rate):
            super().__init__()
            # 使用 ModuleList 存放若干个 Bottleneck 层
            self.layers = nn.ModuleList()
            current_channels = in_channels
            for i in range(n_layers):
                self.layers.append(Bottleneck(current_channels, growth_rate))
                current_channels += growth_rate  # 每层增加 growth_rate 个通道

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class Transition(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            # Transition 层用于压缩通道数并下采样（AvgPool2d）
            self.bn = nn.BatchNorm2d(in_channels)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.pool = nn.AvgPool2d(2)

        def forward(self, x):
            out = F.relu(self.bn(x))
            out = self.conv(out)
            return self.pool(out)

    class DenseNet_CIFAR(nn.Module):
        def __init__(self, block_config=(4, 4, 4, 4), growth_rate=16, num_classes=10):
            super().__init__()
            # 初始通道数通常为 2*growth_rate
            num_channels = 2 * growth_rate
            self.conv0 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)

            self.dense_blocks = nn.ModuleList()
            self.transitions = nn.ModuleList()

            in_channels = num_channels
            # 根据 block_config 逐个构建 DenseBlock 与 Transition
            for i, num_layers in enumerate(block_config):
                block = DenseBlock(num_layers, in_channels, growth_rate)
                self.dense_blocks.append(block)
                in_channels += num_layers * growth_rate
                if i != len(block_config) - 1:
                    # 除最后一个 block 外，每段后面加一个 Transition 压缩通道
                    out_channels = in_channels // 2
                    self.transitions.append(Transition(in_channels, out_channels))
                    in_channels = out_channels

            self.final_bn = nn.BatchNorm2d(in_channels)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # 分类器：先 dropout 再线性分类
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_channels, num_classes)
            )

        def forward(self, x):
            out = self.conv0(x)
            for i, block in enumerate(self.dense_blocks):
                out = block(out)
                if i < len(self.transitions):
                    out = self.transitions[i](out)
            out = F.relu(self.final_bn(out))
            out = self.avgpool(out)
            out = torch.flatten(out, 1)  # 展平为 (batch, features)
            out = self.classifier(out)
            return out

    # 实例化模型并移动到GPU
    model = DenseNet_CIFAR(block_config=(4, 4, 4, 4), growth_rate=16, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    # 优化器中 weight_decay 即 L2 正则化系数，用于控制权重大小（防止过拟合）
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # 余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

    return model, criterion, optimizer, scheduler, trainloader, testloader, device

def draw_loss_and_accuracy_curve(loss, steps, train_acc, test_acc, save_path):
    # 绘制训练损失与训练/测试准确率曲线，训练损失使用左 y 轴，准确率使用右 y 轴
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(steps, loss, 'b-', label='Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(steps, train_acc, 'r-', label='Train Accuracy')
    ax2.plot(steps, test_acc, 'g-', label='Test Accuracy')
    ax2.set_ylabel('Accuracy (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 合并两个图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 标题显示总 epoch 数
    plt.title(f'Training Loss and Accuracy over {len(steps)} Epochs')
    fig.tight_layout()
    plt.savefig(f"{save_path}/loss_and_accuracy_curve.png", dpi=150)
    plt.close()

def predict(test_loader, model, device):
    # 在测试集上评估模型准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # 返回百分比形式的准确率
    return 100.0 * correct / total

def train(train_loader, test_loader, model, epochs, crit, opti, scheduler, save_path, device):
    import os
    os.makedirs(save_path, exist_ok=True)

    # 是否使用 AMP（仅在 CUDA 可用时）
    use_amp = (device.type == 'cuda')
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')

    print("Start Training...")
    epochs_list = []
    losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0

        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opti.zero_grad()

            if use_amp:
                # 混合精度训练代码路径
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = crit(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(opti)
                scaler.update()
            else:
                # 标准训练路径
                outputs = model(inputs)
                loss = crit(outputs, labels)
                loss.backward()
                opti.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 每轮结束后更新学习率调度器
        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        # 在测试集上计算准确率
        test_acc = predict(test_loader, model, device)

        # 保存最优模型与定期保存（每20轮）
        if epoch == 0 or test_acc > max(test_accuracies, default=0):
            torch.save(model.state_dict(), f"{save_path}/best_model.pth")
        if (epoch + 1) % 20 == 0:  # 每20轮保存一次
            torch.save(model.state_dict(), f"{save_path}/epoch_{epoch+1}.pth")

        # 每轮打印训练摘要
        print(f'轮数: {epoch + 1}, 平均损失: {avg_loss:.3f}, 训练准确率: {train_acc:.2f}%, 测试准确率: {test_acc:.2f}%')

        epochs_list.append(epoch + 1)
        losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    print('Finished Training')
    return epochs_list, losses, train_accuracies, test_accuracies

if __name__ == "__main__":

    set_seed(42)

    # 在 Windows 下使用 spawn 启动进程以避免 DataLoader 相关问题
    torch.multiprocessing.set_start_method('spawn', force=True)

    model_save_path = './results_densenet/models'
    plot_save_path = './results_densenet/plots'
    data_save_path = './data'
    if not exists(model_save_path):
        makedirs(model_save_path)
    if not exists(plot_save_path):
        makedirs(plot_save_path)
    if not exists(data_save_path):
        makedirs(data_save_path)

    # 初始化并训练模型
    model, criterion, optimizer, scheduler, trainLoader, testLoader, device = initialization()
    num_epochs = 100
    epochsList, lossesList, trainAccuracy, testAccuracy = train(
        trainLoader, testLoader, model, num_epochs, criterion, optimizer, scheduler, model_save_path, device
    )
    # 绘制并保存损失与准确率曲线图
    draw_loss_and_accuracy_curve(
        lossesList, epochsList, trainAccuracy, testAccuracy, plot_save_path
    )
