import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

# 初始化MediaPipe模型（调低置信度阈值）
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        # 提取关键点（示例：仅使用姿态关键点）
        frame_feature = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                frame_feature.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

        # 如果未检测到关键点，跳过该帧
        if len(frame_feature) > 0:
            frame_features.append(frame_feature)

    cap.release()
    return np.array(frame_features)


def parse_xgtf(xgtf_path):
    tree = ET.parse(xgtf_path)
    root = tree.getroot()
    action = root.find(".//attribute[@name='action']")
    return action.text if action is not None else "unknown"


def load_dataset(annotation_dir, video_dir):
    X, y = [], []
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mpg', '.mp4', '.avi'))]

    for video_file in tqdm(video_files):
        video_path = os.path.join(video_dir, video_file)
        base_name = os.path.splitext(video_file)[0]
        xgtf_path = os.path.join(annotation_dir, f"{base_name}.xgtf")

        if not os.path.exists(xgtf_path):
            continue

        try:
            label = parse_xgtf(xgtf_path)
            features = process_video(video_path)

            if len(features) == 0:
                continue

            video_feature = np.concatenate([np.mean(features, axis=0), np.std(features, axis=0)])
            X.append(video_feature)
            y.append(label)
        except:
            continue

    return np.array(X), np.array(y)


# 配置路径
ANNOTATION_DIR = r"E:\train\bike\train\labels"
VIDEO_DIR = r"E:\train\bike\train\videos"

# 加载数据
X, y = load_dataset(ANNOTATION_DIR, VIDEO_DIR)
print(f"成功加载 {len(X)} 个样本")

# 标签编码（转换为数值）
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 划分数据集
if len(X) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
else:
    print("错误：未加载到任何数据！")
# 在代码末尾添加（建议放在 train_test_split 之后）
if len(X) > 0:
    # 保存特征和标签
    np.save("X_features.npy", X)
    np.save("y_labels.npy", y)

    # 如果需要保存训练集/测试集
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)

    # 保存标签编码器（用于后续预测时解码）
    import joblib

    joblib.dump(encoder, "label_encoder.pkl")

    print("数据已保存到当前工作目录")