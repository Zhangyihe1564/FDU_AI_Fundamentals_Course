import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

show = ToPILImage()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size,
                                          shuffle=True, num_workers=0, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size,
                                         shuffle=False, num_workers=0, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train(train_loader, model, num_of_epochs, Criterion, Optimizer,
          path_to_save, draw_loss=None,draw_steps=None,step_interval=2000):
    if draw_loss is None:
        draw_loss = []
    if draw_steps is None:
        draw_steps = []

    global_step = 0
    for epoch in range(num_of_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            global_step += 1
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            Optimizer.zero_grad()
            outputs = model(inputs)
            loss_func = Criterion(outputs, labels)
            loss_func.backward()
            Optimizer.step()

            running_loss += loss_func.item()
            if global_step % step_interval == 0:
                avg = running_loss / step_interval
                print('epoch %d: global_step %6d loss: %.3f' % (epoch + 1, global_step, avg))
                draw_loss.append(avg)
                draw_steps.append(global_step)
                running_loss = 0.0

        torch.save(model.state_dict(), f"{path_to_save}/epoch_{epoch + 1}_model.pth")

    print('Finished Training')
    return draw_loss, draw_steps

def predict(test_loader, model):
    model.to(device)
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

    acc = 100.0 * correct / total if total > 0 else 0.0
    print('测试集中的准确率为: %.2f %%' % acc)
    return acc
