import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from .A_dataset import ImageTxtDataset
from .B_myalex import alex

# 准备数据集
train_data = ImageTxtDataset(r"F:\新建文件夹\华清远见\day3\dataset\train.txt",
                             r"F:\新建文件夹\华清远见\day3\dataset\Images\train",
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                                 ])
                             )

test_data = ImageTxtDataset(r"F:\新建文件夹\华清远见\day3\dataset\val.txt",
                            r"F:\新建文件夹\华清远见\day3\dataset\Images\val",
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                                ])
                            )

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度: {train_data_size}")
print(f"测试数据集的长度: {test_data_size}")

# 加载数据集（使用 `num_workers` 提高数据加载速度）
train_loader = DataLoader(train_data, batch_size=128, num_workers=0, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, num_workers=0)

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建模型并移动到 GPU
chen = alex().to(device)

# 创建损失函数并移动到 GPU
loss_fn = nn.CrossEntropyLoss().to(device)

# 优化器 (使用 Adam 替换 SGD)
learning_rate = 0.01
optim = torch.optim.Adam(chen.parameters(), lr=learning_rate)

# 设置训练参数
total_train_step = 0
total_test_step = 0
epoch = 10

# 添加 tensorboard
writer = SummaryWriter("../../logs_train")

# 记录开始时间
start_time = time.time()

for i in range(epoch):
    print(f"-----第 {i + 1} 轮训练开始-----")

    # 训练步骤
    chen.train()
    train_bar = tqdm(train_loader, desc=f"训练进度 (轮数 {i + 1}/{epoch})")  # 训练进度条
    for imgs, targets in train_bar:
        imgs, targets = imgs.to(device), targets.to(device)

        outputs = chen(imgs)
        loss = loss_fn(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step += 1
        train_bar.set_postfix(loss=loss.item())  # 在进度条中显示 loss

        if total_train_step % 500 == 0:
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    chen.eval()
    total_test_loss = 0.0
    total_accuracy = 0
    test_bar = tqdm(test_loader, desc="测试进度")  # 测试进度条
    with torch.no_grad():
        for imgs, targets in test_bar:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = chen(imgs)

            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

            test_bar.set_postfix(loss=total_test_loss, accuracy=total_accuracy.item() / test_data_size)

    print(f"测试集 loss: {total_test_loss}")
    print(f"测试集正确率: {total_accuracy / test_data_size:.4f}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    # 保存模型
    torch.save(chen.state_dict(), f"model_save/chen_{i}.pth")
    print("模型已保存")

writer.close()
