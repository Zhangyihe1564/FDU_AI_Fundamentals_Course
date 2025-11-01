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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证 CUDA 算子确定性（但可能降低速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ✅ 修正：使用标准 CIFAR-10 的均值和标准差
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)  # 原始论文和 torchvision 推荐值

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # ✅ 修正：Bottleneck 正确实现（保持不变，这部分没问题）
    class Bottleneck(nn.Module):
        def __init__(self, in_channels, growth_rate):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(4 * growth_rate)
            self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

        def forward(self, x):
            out = F.relu(self.bn1(x))
            out = F.relu(self.bn2(self.conv1(out)))
            out = self.conv2(out)
            return torch.cat([x, out], 1)

    class DenseBlock(nn.Module):
        def __init__(self, n_layers, in_channels, growth_rate):
            super().__init__()
            self.layers = nn.ModuleList()
            current_channels = in_channels
            for i in range(n_layers):
                self.layers.append(Bottleneck(current_channels, growth_rate))
                current_channels += growth_rate  # 每层输出增加 growth_rate 通道

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)  # x 自动累积通道
            return x

    class Transition(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.bn = nn.BatchNorm2d(in_channels)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.pool = nn.AvgPool2d(2)

        def forward(self, x):
            out = F.relu(self.bn(x))
            out = self.conv(out)
            return self.pool(out)

    class DenseNet_CIFAR(nn.Module):
        def __init__(self, block_config=(6, 6, 6, 6), growth_rate=12, num_classes=10):
            super().__init__()
            num_channels = 2 * growth_rate
            self.conv0 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)

            self.dense_blocks = nn.ModuleList()
            self.transitions = nn.ModuleList()

            in_channels = num_channels
            for i, num_layers in enumerate(block_config):
                block = DenseBlock(num_layers, in_channels, growth_rate)
                self.dense_blocks.append(block)
                in_channels += num_layers * growth_rate
                if i != len(block_config) - 1:
                    out_channels = in_channels // 2
                    self.transitions.append(Transition(in_channels, out_channels))
                    in_channels = out_channels

            self.final_bn = nn.BatchNorm2d(in_channels)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
            out = torch.flatten(out, 1)
            out = self.classifier(out)
            return out

    # ----------------------------
    # 模型、损失、优化器
    # ----------------------------
    model = DenseNet_CIFAR(block_config=(6, 6, 6, 6), growth_rate=12, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

    return model, criterion, optimizer, scheduler, trainloader, testloader, device

def draw_loss_and_accuracy_curve(loss, steps, train_acc, test_acc, save_path):
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

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f'Training Loss and Accuracy over {len(steps)} Epochs')
    fig.tight_layout()
    plt.savefig(f"{save_path}/loss_and_accuracy_curve.png", dpi=150)
    plt.close()

def predict(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # 移除 AMP：推理不需要，且避免设备绑定
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # ✅ 转为 Python 数值
    return 100.0 * correct / total  # ✅ 直接返回 float

def train(train_loader, test_loader, model, epochs, crit, opti, scheduler, save_path, device):
    import os
    os.makedirs(save_path, exist_ok=True)

    # ✅ 动态启用 AMP
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
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = crit(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(opti)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = crit(outputs, labels)
                loss.backward()
                opti.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # ✅ 关键：更新学习率！
        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        test_acc = predict(test_loader, model, device)

        # ✅ 只保存最佳模型（或定期保存）
        if epoch == 0 or test_acc > max(test_accuracies, default=0):
            torch.save(model.state_dict(), f"{save_path}/best_model.pth")
        if (epoch + 1) % 20 == 0:  # 每20轮保存一次
            torch.save(model.state_dict(), f"{save_path}/epoch_{epoch+1}.pth")

        print(f'轮数: {epoch + 1}, 平均损失: {avg_loss:.3f}, 训练准确率: {train_acc:.2f}%, 测试准确率: {test_acc:.2f}%')

        epochs_list.append(epoch + 1)
        losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    print('Finished Training')
    return epochs_list, losses, train_accuracies, test_accuracies

if __name__ == "__main__":

    set_seed(42)

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


    model, criterion, optimizer, scheduler, trainLoader, testLoader, device = initialization()
    num_epochs = 100
    epochsList, lossesList, trainAccuracy, testAccuracy = train(
        trainLoader, testLoader, model, num_epochs, criterion, optimizer, scheduler, model_save_path, device
    )
    draw_loss_and_accuracy_curve(
        lossesList, epochsList, trainAccuracy, testAccuracy, plot_save_path
    )