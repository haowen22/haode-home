import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
import warnings

warnings.filterwarnings("ignore")
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)
import mediapipe as mp
import math
import csv
import matplotlib.pyplot as plt
import cv2
from video_path import connect_to_mysql, extract_video_data

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 定义关键关节点的索引
KEY_POINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


# 计算两个向量之间的角度
def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))
    angle = math.degrees(math.acos(cosine_angle))
    return angle


# 绘制关节角度
def draw_angle(frame, landmarks, start_idx, mid_idx, end_idx, joint_name):
    if (landmarks[start_idx].visibility > 0.5 and
            landmarks[mid_idx].visibility > 0.5 and
            landmarks[end_idx].visibility > 0.5):
        start_point = (int(landmarks[start_idx].x * width), int(landmarks[start_idx].y * height))
        mid_point = (int(landmarks[mid_idx].x * width), int(landmarks[mid_idx].y * height))
        end_point = (int(landmarks[end_idx].x * width), int(landmarks[end_idx].y * height))

        angle = calculate_angle(start_point, mid_point, end_point)
        cv2.putText(frame, f"{joint_name}: {int(angle)}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return angle
    return None


# 连接数据库并获取视频路径
conn = connect_to_mysql()
if not conn:
    print("无法连接到数据库，程序退出")
    exit()

try:
    video_path = extract_video_data(conn)
    print(f"🎉 成功获取视频文件: {video_path}")

    if not os.path.exists(video_path):
        print(f"错误：文件 '{video_path}' 不存在")
        exit()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 '{video_path}'")
        exit()

    # 获取视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = r"E:\innovation contest\study\game\med1\output_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 定义关节角度信息
    JOINT_ANGLES = [
        (11, 13, 15, "Left Elbow"),
        (12, 14, 16, "Right Elbow"),
        (23, 25, 27, "Left Knee"),
        (24, 26, 28, "Right Knee"),
        (13, 11, 23, "Left Shoulder"),
        (14, 12, 24, "Right Shoulder"),
        (11, 23, 25, "Left Hip"),
        (12, 24, 26, "Right Hip")
    ]

    # 初始化角度数据存储
    angles_over_time = {joint_name: [] for _, _, _, joint_name in JOINT_ANGLES}

    # 处理视频
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # 绘制关节点
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if idx in KEY_POINTS:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # 绘制连接线
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
                (25, 27), (26, 28)
            ]
            for start_idx, end_idx in connections:
                if start_idx in KEY_POINTS and end_idx in KEY_POINTS:
                    start_point = (int(results.pose_landmarks.landmark[start_idx].x * width),
                                   int(results.pose_landmarks.landmark[start_idx].y * height))
                    end_point = (int(results.pose_landmarks.landmark[end_idx].x * width),
                                 int(results.pose_landmarks.landmark[end_idx].y * height))
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

            # 计算并记录关节角度
            for start_idx, mid_idx, end_idx, joint_name in JOINT_ANGLES:
                angle = draw_angle(frame, results.pose_landmarks.landmark, start_idx, mid_idx, end_idx, joint_name)
                if angle is not None:
                    angles_over_time[joint_name].append(angle)

        cv2.imshow("Pose Estimation", frame)
        out.write(frame)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 分析关节角度
    IDEAL_ANGLES = {
        "Left Elbow": (120, 160),
        "Right Elbow": (120, 160),
        "Left Knee": (140, 170),
        "Right Knee": (140, 170),
        "Left Shoulder": (80, 120),
        "Right Shoulder": (80, 120),
        "Left Hip": (70, 110),
        "Right Hip": (70, 110)
    }

    analysis_result = {}
    for joint, angles in angles_over_time.items():
        errors = []
        ideal_range = IDEAL_ANGLES.get(joint, (0, 180))
        outliers = [i for i, ang in enumerate(angles) if not (ideal_range[0] <= ang <= ideal_range[1])]

        if outliers:
            errors.append(f"检测到{len(outliers)}帧角度超出理想范围{ideal_range}")

        analysis_result[joint] = {
            "ideal_range": ideal_range,
            "errors": errors,
            "outlier_frames": outliers
        }

    # 打印分析报告
    print("\n骑行姿势分析报告：")
    for joint, data in analysis_result.items():
        print(f"\n[{joint}]")
        print(f"理想角度范围：{data['ideal_range']}")
        if data["errors"]:
            print("检测到问题：")
            print("\n".join(data["errors"]))
        else:
            print("角度变化在正常范围内")

    # 可视化
    plt.figure(figsize=(15, 10))
    for idx, (joint, angles) in enumerate(angles_over_time.items(), 1):
        plt.subplot(4, 2, idx)
        plt.plot(angles, label='实际角度', color='blue', alpha=0.6)
        ideal = analysis_result[joint]["ideal_range"]
        plt.axhspan(ideal[0], ideal[1], color='green', alpha=0.2, label='理想范围')
        outliers = analysis_result[joint]["outlier_frames"]
        plt.scatter(outliers, [angles[i] for i in outliers], color='red', s=20, label='异常角度')
        plt.title(joint)
        plt.xlabel("帧数")
        plt.ylabel("角度(度)")
        plt.legend()
        error_text = "\n".join(analysis_result[joint]["errors"])
        if error_text:
            plt.annotate(error_text, xy=(0.5, 0.1), xycoords='axes fraction',
                         ha='center', color='red', bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.show()

    # 保存数据到CSV
    csv_path = os.path.join(os.path.expanduser("~"), "Desktop", "骑行矫正文件.csv")
    try:
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Frame", "Joint", "Angle"])
            for frame, angles in enumerate(zip(*angles_over_time.values())):
                for joint_name, angle in zip(angles_over_time.keys(), angles):
                    writer.writerow([frame, joint_name, angle])
        print(f"数据已保存到：{csv_path}")
    except PermissionError:
        print(f"错误：无法保存文件，请检查路径权限：{csv_path}")

except Exception as e:
    print(f"处理过程中发生错误: {str(e)}")
finally:
    if 'conn' in locals() and conn.is_connected():
        conn.close()