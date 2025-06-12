import time
import torch
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

# 定义训练的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理（调整尺寸等，ResNet 对输入尺寸无 AlexNet 那样严格 227×227 要求，这里用常用的 32×32 也可，不过 ResNet 默认适配 ImageNet 相关尺寸，实际可灵活调整，CIFAR10 原始是 32×32，这里用 transforms.Resize 调整到 224×224 更贴合 ResNet 训练习惯  ）
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化，符合 ImageNet 数据分布习惯，提升训练效果
])

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./model_save",
                                          train=True,
                                          transform=transform,
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root="./model_save",
                                         train=False,
                                         transform=transform,
                                         download=True)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度{train_data_size}")
print(f"测试数据集的长度{test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 创建 ResNet 模型（这里以 ResNet18 为例，可替换成 ResNet34、ResNet50 等，需要注意输入通道和分类数  ）
model = torchvision.models.resnet18(pretrained=False)  # pretrained=False 表示不加载预训练权重，若要加载预训练权重设为 True，后续还需调整全连接层
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR10 是 10 分类，修改最后全连接层输出维度
model = model.to(device)

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # SGD 优化器，带动量

# 设置训练网络的一些参数
total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录测试的次数
epochs = 10  # 训练的轮数

# 创建保存训练结果的目录
os.makedirs("training_results/logs_train", exist_ok=True)
os.makedirs("training_results/model_save", exist_ok=True)

# 添加 tensorboard
writer = SummaryWriter("training_results/logs_train")

# 添加开始时间
start_time = time.time()

for epoch in range(epochs):
    print(f"-----第{epoch + 1}轮训练开始-----")
    # 训练步骤
    model.train()
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_train_step += 1
        if total_train_step % 200 == 0:
            print(f"第{total_train_step}步的训练损失: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤（以测试数据上的正确率来评估模型）
    model.eval()
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的损失: {total_test_loss}")
    test_accuracy = total_accuracy / test_data_size
    print(f"整体测试集上的正确率：{test_accuracy}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", test_accuracy, total_test_step)
    total_test_step += 1

    # 保存每一轮训练模型
    torch.save(model, f"training_results/model_save/resnet_epoch_{epoch}.pth")
    print("当前轮次模型已保存")

end_time = time.time()
print(f"总训练时间: {end_time - start_time} 秒")
writer.close()