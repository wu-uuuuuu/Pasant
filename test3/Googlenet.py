import time
import torch
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

# 定义训练设备（自动选择GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理（GoogLeNet要求输入尺寸为224×224）
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),  # 调整为GoogLeNet标准输入尺寸
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准归一化
])

# 加载CIFAR10数据集
train_data = torchvision.datasets.CIFAR10(
    root="./model_save", train=True, transform=transform, download=True
)
test_data = torchvision.datasets.CIFAR10(
    root="./model_save", train=False, transform=transform, download=True
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集长度: {train_data_size}")
print(f"测试数据集长度: {test_data_size}")

# 创建GoogLeNet模型（修改分类层以适配CIFAR10的10分类）
model = torchvision.models.googlenet(pretrained=False, aux_logits=False)  # 禁用辅助分类器（可选）
model.fc = nn.Linear(model.fc.in_features, 10)  # 修改最后一层为10分类
model = model.to(device)

# 损失函数与优化器
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练参数
total_train_step = 0
total_test_step = 0
epochs = 10

# 创建保存目录
os.makedirs("training_results1/logs_train", exist_ok=True)
os.makedirs("training_results1/model_save", exist_ok=True)

# TensorBoard日志
writer = SummaryWriter("training_results/logs_train")

# 开始训练
start_time = time.time()

for epoch in range(epochs):
    print(f"-----第{epoch + 1}轮训练开始-----")

    # 训练阶段
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
    print(f"测试集 loss: {total_test_loss}, 准确率: {test_accuracy}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", test_accuracy, total_test_step)
    total_test_step += 1

    # 保存模型
    torch.save(model, f"training_results1/model_save/googlenet_epoch_{epoch}.pth")
    print("当前轮次模型已保存")

end_time = time.time()
print(f"总训练时间: {end_time - start_time} 秒")
writer.close()