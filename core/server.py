import os
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS

from model import mobilenetv3_large, mobilenetv3_small

# MTCNN 人脸检测器初始化
detector = MTCNN()

# 超参数配置
config = {
    "model_type": "large",  # 可以选择 "large" 或 "small"
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_path": "./model/best_model.pth",  # 保存最佳模型的路径
}

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 函数：对齐、裁剪、调整大小
def align_face(img, keypoints):
    # 获取左眼和右眼的坐标
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # 计算眼睛的角度
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    # 计算旋转角度
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(eye_center, angle, 1)

    # 获取对齐后的图像
    aligned_face = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

    return aligned_face


# 函数：检测人脸、对齐、裁剪
def detect_align_crop(img):
    # 将图片转换为RGB，因为OpenCV读取的是BGR格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 检测人脸
    faces = detector.detect_faces(img_rgb)

    if len(faces) == 0:
        print("No face detected in the image.")
        return None

    # 获取人脸框和关键点
    face = faces[0]
    keypoints = face['keypoints']

    # 对齐人脸
    aligned_face = align_face(img_rgb, keypoints)

    # 获取人脸框坐标并裁剪
    x, y, w, h = face['box']
    face_cropped = aligned_face[y:y + h, x:x + w]

    # 调整大小为112x112
    face_resized = cv2.resize(face_cropped, (112, 112))

    return face_resized


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


def extract_vector(image, feature_extractor):
    try:
        # 将OpenCV的BGR格式转换为PIL的RGB格式
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = transform(image).unsqueeze(0).to(config["device"])
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

    # 提取图片的嵌入
    with torch.no_grad():
        embedding = feature_extractor(image)
        # 将嵌入展平
        embedding = embedding.view(embedding.size(0), -1)
        embedding = embedding.cpu().numpy()
    return embedding


app = Flask(__name__)
CORS(app)  # 启用跨域支持

feature_extractor = load_model()


@app.route('/embedding', methods=['POST'])
def extract_embedding():
    print('接收到的请求文件:', request.files)  # 打印请求文件信息
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    try:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        face = detect_align_crop(img)
        if face is None:
            return jsonify({"error": "No face detected in the image"}), 400
        vector = extract_vector(face, feature_extractor)
        vector = vector[0]
        if vector is None:
            return jsonify({"error": "Error extracting embedding"}), 500
        return jsonify({"embedding": vector.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
