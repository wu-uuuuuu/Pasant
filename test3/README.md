#训练自己的数据集指南
##一、数据集准备与预处理
###1.1 数据收集与组织

目录结构：按类别划分训练集和验证集

###1.2 数据增强（Data Augmentation）

通过随机变换增加训练样本多样性，提高模型泛化能力：

```
import torchvision.transforms as transforms
```

#### 训练集增强
```
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪并调整大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),      # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
    transforms.ToTensor(),              # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
])
```

#### 验证集仅需调整大小和标准化
```
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

###1.3 数据集加载

使用 PyTorch 的ImageFolder加载图像数据：

```
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
```

#### 加载数据集
```
train_dataset = ImageFolder(root='dataset/train', transform=train_transform)
val_dataset = ImageFolder(root='dataset/val', transform=val_transform)
```
#### 创建数据加载器
```
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
```

#### 查看类别信息
```
class_names = train_dataset.classes
num_classes = len(class_names)
print(f"类别: {class_names}")
print(f"类别数量: {num_classes}")
```

##二、模型构建与微调
###2.1 迁移学习（Transfer Learning）
使用预训练模型（如 ResNet50）进行微调：

```
import torch
import torch.nn as nn
import torchvision.models as models
```

#### 加载预训练模型
```
model = models.resnet50(pretrained=True)
```

#### 冻结特征提取层（可选）
```
for param in model.parameters():
    param.requires_grad = False
```

#### 修改分类层以适应自定义类别数量
```
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # 替换最后的全连接层
```

#### 移至GPU（如果可用）
```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```
###2.2 损失函数与优化器
针对分类任务，使用交叉熵损失和 SGD 优化器：
```
criterion = nn.CrossEntropyLoss()
```
#### 仅优化新添加的分类层参数（如果冻结了特征层）
```
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
```
#### 学习率调度器（每7个epoch降低学习率）
```
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
```
##三、训练与验证
###3.1 训练循环
实现完整的训练和验证流程：

```
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 每个epoch有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
                dataloader = train_loader
            else:
                model.eval()   # 评估模式
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # 迭代数据
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 梯度归零
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 如果是训练阶段，反向传播+优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler:
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Best val Acc: {best_acc:.4f}')
    return model
```
#### 开始训练
```
trained_model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
```
###3.2 训练技巧
早停法（Early Stopping）：当验证损失不再下降时提前结束训练

学习率调整：使用ReduceLROnPlateau动态降低学习率
```
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
# 在验证循环后调用
scheduler.step(val_loss)
```
##四、模型评估与可视化
###4.1 混淆矩阵与分类报告
使用 scikit-learn 评估模型性能：

```
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 打印分类报告
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png')
    plt.show()

# 评估验证集
evaluate_model(trained_model, val_loader)
```
###4.2 学习曲线可视化
跟踪训练和验证过程中的损失和准确率：

```
def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('训练和验证损失')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('训练和验证准确率')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
```
##五、模型部署与推理
###5.1 保存与加载模型
```
# 保存完整模型
torch.save(model, 'full_model.pth')

# 加载模型用于推理
loaded_model = torch.load('full_model.pth')
loaded_model.eval()  # 设置为评估模式
```
###5.2 单张图像推理
```
from PIL import Image

def predict_image(image_path, model, transform, class_names):
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    
    # 获取预测结果
    predicted_class = class_names[preds[0]]
    confidence = probs[0][preds[0]].item()
    
    return predicted_class, confidence

# 示例
image_path = 'test_image.jpg'
pred_class, pred_conf = predict_image(image_path, loaded_model, val_transform, class_names)
print(f'预测类别: {pred_class}, 置信度: {pred_conf:.4f}')
```
##六、常见问题与解决方案
###6.1 过拟合问题
表现：训练准确率高，验证准确率低

解决方案：

(1)增加数据增强

(2)添加 Dropout 层

(3)减小模型复杂度

(4)应用 L2 正则化（weight decay）
```
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
```
###6.2 欠拟合问题
表现：训练和验证准确率都低

解决方案：

(1)增加模型复杂度

(2)延长训练时间

(3)调整学习率

尝试不同的网络架构
###6.3 数据不平衡
表现：某些类别样本远多于其他类别

解决方案：

(1)过采样少数类

(2)欠采样多数类

(3)使用类别权重

```
# 计算类别权重
class_counts = [len(os.listdir(f'dataset/train/{cls}')) for cls in class_names]
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum()

# 应用到损失函数
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```


##七、总结与扩展
核心步骤：数据准备 → 模型构建 → 训练调优 → 评估部署

扩展方向：

目标检测（Faster R-CNN、YOLO）

语义分割（U-Net、DeepLab）

目标跟踪（DeepSORT）

模型量化与部署（TensorRT、ONNX）