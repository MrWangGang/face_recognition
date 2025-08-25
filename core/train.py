import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import mobilenetv3_large, mobilenetv3_small

# 超参数配置
config = {
    "model_type": "large",  # 选择 "large" 或 "small"
    "data_dir": "./datasets/train_cropped",  # 训练数据目录
    "batch_size": 64,
    "num_epochs": 150,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "train_val_split": 0.8,  # 80% 作为训练集，20% 作为验证集
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_path": "./model/best_model.pth"  # 保存最佳模型的路径
}

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU，还需要设置
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(2035)

# 数据预处理 - 训练集
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(112),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 数据预处理 - 验证集
test_transforms = transforms.Compose([
    transforms.Resize(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
full_dataset = datasets.ImageFolder(config["data_dir"], transform=train_transforms)

# 获取类别信息
class_names = full_dataset.classes
num_classes = len(class_names)

# 根据 train_val_split 比例随机选择 20% 样本作为验证集
train_size = len(full_dataset)  # 保持整个数据集作为训练集
val_size = int(0.2 * len(full_dataset))  # 随机选择 20% 作为验证集

# 从训练集中随机选择 20% 样本作为验证集
train_dataset = full_dataset
val_dataset, _ = random_split(train_dataset, [val_size, len(full_dataset) - val_size])

# 将整个训练集加载为训练集，验证集是从训练集中分割出来的
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

# 加载预训练模型权重
pretrained_weights_path = './model/mobilenetv3-large-pre.pth'  # 请替换为你实际的预训练模型路径
try:
    pretrained_state_dict = torch.load(pretrained_weights_path, map_location=config["device"], weights_only=True)
except FileNotFoundError:
    print(f"Error: The pre-trained model file {pretrained_weights_path} was not found.")
    import sys
    sys.exit(1)

# 初始化你的模型
if config["model_type"] == "large":
    model = mobilenetv3_large(num_classes=1000)  # 预训练模型可能是在 1000 类上训练的，先以此初始化
else:
    model = mobilenetv3_small(num_classes=1000)

# 获取模型的 state_dict
model_state_dict = model.state_dict()

# 调整预训练权重的键名以匹配你的模型
new_pretrained_state_dict = {}
for k, v in pretrained_state_dict.items():
    if k in model_state_dict:
        new_pretrained_state_dict[k] = v

# 加载预训练权重到你的模型
model.load_state_dict(new_pretrained_state_dict, strict=False)

# 根据新任务调整分类器（假设新任务有 num_classes 个类别）
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
model = model.to(config["device"])

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# 将 max_val_accuracy 初始化移到训练循环外部
max_train_accuracy = 0

# 开始训练周期，遍历设定的训练轮数
for epoch in range(config["num_epochs"]):
    model.train()  # 设置模型为训练模式

    # 初始化用于累计的变量：运行损失、正确分类的训练样本数和总训练样本数
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # 初始化训练阶段的进度条
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} [Training]", leave=False)

    # 遍历训练数据集
    for images, labels in train_pbar:
        images, labels = images.to(config["device"]), labels.to(config["device"])  # 将图像和标签数据移至指定设备

        optimizer.zero_grad()  # 清空之前的梯度

        outputs = model(images)  # 获取模型的预测输出
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播求梯度
        optimizer.step()  # 根据梯度更新模型参数

        # 累计损失和正确分类的数量
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_pbar.close()  # 关闭训练阶段的进度条

    # 计算训练准确率
    train_accuracy = 100 * correct_train / total_train

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config["device"]), labels.to(config["device"])
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # 计算验证准确率
    val_accuracy = 100 * correct_val / total_val

    # 打印每轮的训练损失、验证损失和验证准确率
    print(f'\nEpoch {epoch + 1}, Train Loss: {running_loss / len(train_loader)}, '
          f'Val Loss: {val_loss / len(val_loader)}, Accuracy: {val_accuracy}%')

    # 保存最佳模型
    if val_accuracy > max_train_accuracy:
        max_train_accuracy = val_accuracy
        torch.save(model.state_dict(), config['checkpoint_path'])  # 保存表现最好的模型
        print(f"saving model for acc {val_accuracy:.2f}%")
