import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 导入定义好的 MobileNetV3 模型
from model import mobilenetv3_large, mobilenetv3_small  # 请将 your_module 替换为实际的模块名

# 超参数配置
config = {
    "model_type": "large",  # 可以选择 "large" 或 "small"
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_path": "./model/best_model.pth",  # 保存最佳模型的路径
    "image_path_1": "./1.jpg",  # 第一张图片路径，需替换为实际路径
    "image_path_2": "./2.jpeg"  # 第二张图片路径，需替换为实际路径
}

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    # 初始化模型
    if config["model_type"] == "large":
        model = mobilenetv3_large()
    else:
        model = mobilenetv3_small()

    # 加载训练好的模型权重
    try:
        state_dict = torch.load(config["checkpoint_path"], map_location=config["device"])
        # 过滤掉 classifier 层的参数
        filtered_state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        model.load_state_dict(filtered_state_dict, strict=False)
    except FileNotFoundError:
        print(f"Error: The trained model file {config['checkpoint_path']} was not found.")
        import sys
        sys.exit(1)

    # 将模型移动到指定设备
    model = model.to(config["device"])

    # 去掉模型的分类器部分，只保留特征提取层
    if config["model_type"] == "large":
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
    else:
        feature_extractor = nn.Sequential(*list(model.children())[:-1])

    # 设置模型为评估模式
    feature_extractor.eval()
    return feature_extractor


def extract_vector(image_path, feature_extractor):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(config["device"])
    except FileNotFoundError:
        print(f"Error: The image file {image_path} was not found.")
        import sys
        sys.exit(1)

    # 提取图片的嵌入
    with torch.no_grad():
        embedding = feature_extractor(image)
        # 将嵌入展平
        embedding = embedding.view(embedding.size(0), -1)
        embedding = embedding.cpu().numpy()
    return embedding


if __name__ == "__main__":
    feature_extractor = load_model()

    # 提取第一张图片的特征向量
    vector_1 = extract_vector(config["image_path_1"], feature_extractor)

    # 提取第二张图片的特征向量
    vector_2 = extract_vector(config["image_path_2"], feature_extractor)

    # 计算余弦相似度
    similarity = cosine_similarity(vector_1, vector_2)[0][0]

    print(f"图片 {config['image_path_1']} 和 {config['image_path_2']} 的余弦相似度为: {similarity}")