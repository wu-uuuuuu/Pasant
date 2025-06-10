import time
import torch
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 定义AlexNet模型
from day2.train_GPU_2 import test_data_size


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出27x27x48（输入227x227）
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出13x13x128
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出6x6x128
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 128 * 6 * 6)  # 展平
        x = self.classifier(x)
        return x

# 定义训练设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理（调整尺寸到227x227，AlexNet输入要求）
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((227, 227)),
    torchvision.transforms.ToTensor(),
])

# 加载数据集
train_data = torchvision.datasets.CIFAR10(
    root="./model_save",
    train=True,
    transform=transform,
    download=True
)
test_data = torchvision.datasets.CIFAR10(
    root="./model_save",
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 初始化模型、损失函数、优化器
model = AlexNet().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # AlexNet常用SGD+momentum

# 训练参数
epochs = 10
best_accuracy = 0.0
total_train_step = 0
total_test_step = 0

# TensorBoard日志保存路径
writer = SummaryWriter("training_results/logs_train")

start_time = time.time()

for epoch in range(epochs):
    print(f"-----第{epoch+1}轮训练开始-----")
    model.train()
    for data in train_loader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 200 == 0:
            print(f"第{total_train_step}步训练 loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试阶段
    model.eval()
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    test_accuracy = total_accuracy / test_data_size
    print(f"测试集 loss: {total_test_loss}, 正确率: {test_accuracy}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", test_accuracy, total_test_step)
    total_test_step += 1

    # 保存最佳模型（根据测试正确率）
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model, "training_results/model_save/best_model.pth")
        print("最佳模型已更新并保存")

    # 保存每轮模型（最终模型可保存最后一轮）
    torch.save(model, f"training_results/model_save/epoch_{epoch}.pth")
    print("当前轮次模型已保存")

end_time = time.time()
print(f"总训练时间: {end_time - start_time} 秒")
writer.close()