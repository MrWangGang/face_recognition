import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# 导入定义好的 MobileNetV3 模型
from model import mobilenetv3_large, mobilenetv3_small  # 请将 your_module 替换为实际的模块名

# 超参数配置
config = {
    "model_type": "large",  # 可以选择 "large" 或 "small"
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_path": "./model/best_model.pth",  # 保存最佳模型的路径
    "image_folder": "./datasets/train_cropped",  # 图片文件夹路径
    "target_image": "./000_0.bmp"  # 目标图片
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


def find_similar_images(target_vector, feature_extractor):
    top_10_similarities = []
    # 遍历类别文件夹
    for class_folder in os.listdir(config["image_folder"]):
        class_path = os.path.join(config["image_folder"], class_folder)
        if os.path.isdir(class_path):
            # 遍历类别的图片
            for filename in os.listdir(class_path):
                if filename.endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_path, filename)
                    if image_path != config["target_image"]:
                        vector = extract_vector(image_path, feature_extractor)
                        similarity = cosine_similarity(target_vector, vector)[0][0]
                        # 将当前图片及其相似度添加到列表中
                        top_10_similarities.append((image_path, similarity))
                        # 保持列表只包含前 10 个最相似的图片
                        top_10_similarities = sorted(top_10_similarities, key=lambda x: x[1], reverse=True)[:10]

                        # 输出当前最相似的前 10 个图片及其相似度
                        print(f"与 {config['target_image']} 最相近的前 10 张图片（当前比较结果）：")
                        for i, (img_path, sim) in enumerate(top_10_similarities, 1):
                            print(f"第 {i} 相似的图片: {img_path}, 相似度: {sim}")
                        print("-" * 50)  # 分隔线
    return top_10_similarities


if __name__ == "__main__":
    feature_extractor = load_model()

    # 提取目标图片的特征向量
    target_vector = extract_vector(config["target_image"], feature_extractor)

    # 找出最相似的 10 张图片
    top_10_similarities = find_similar_images(target_vector, feature_extractor)

    print(f"最终与 {config['target_image']} 最相近的前 10 张图片：")
    for i, (image_path, similarity) in enumerate(top_10_similarities, 1):
        print(f"第 {i} 相似的图片: {image_path}, 相似度: {similarity}")