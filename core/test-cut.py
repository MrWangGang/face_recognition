import os

import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image

# MTCNN 人脸检测器初始化
detector = MTCNN()

# 输入和输出路径
input_image_path = './your_inpput_image.png'  # 输入的单张图片路径，需替换为实际路径
output_image_path = './your_output_image.png'  # 输出的图片路径，需替换为实际路径

# 函数：对齐、裁剪、调整大小并保存到指定路径
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

# 函数：检测人脸、对齐、裁剪并保存到指定路径
def detect_align_crop_save(input_path, output_path):
    # 读取图片
    img = cv2.imread(input_path)

    # 将图片转换为RGB，因为OpenCV读取的是BGR格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 检测人脸
    faces = detector.detect_faces(img_rgb)

    if len(faces) == 0:
        print(f"No face detected in {input_path}")
        return

    # 获取人脸框和关键点
    face = faces[0]
    keypoints = face['keypoints']

    # 对齐人脸
    aligned_face = align_face(img_rgb, keypoints)

    # 获取人脸框坐标并裁剪
    x, y, w, h = face['box']
    face_cropped = aligned_face[y:y+h, x:x+w]

    # 调整大小为224x224
    face_resized = cv2.resize(face_cropped, (112, 112))

    # 将图片转换回BGR并保存
    face_resized_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存图像
    cv2.imwrite(output_path, face_resized_bgr)
    print(f"Image saved to {output_path}")

# 处理并保存单张图片
detect_align_crop_save(input_image_path, output_image_path)