import os
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 彻底禁用 TensorFlow 日志
os.environ['GLOG_minloglevel'] = '3'      # 禁用 MediaPipe 的 glog 日志
import warnings
warnings.filterwarnings("ignore")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import hashlib
import requests
import json
import cv2
import mediapipe as mp
import math
import csv
import os
import time
import gc
import matplotlib.pyplot as plt
from io import BytesIO
import zlib
import base64
from datetime import datetime
import mysql.connector
from testnew import angles, angles_over_time
from video_path import connect_to_mysql, extract_video_data

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class VideoCaptureContext:
    """安全的视频捕获上下文管理器"""

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"无法打开视频文件: {self.video_path}")
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


def safe_delete(file_path, max_retries=3):
    """带重试机制的文件删除"""
    for i in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ 成功删除文件: {file_path}")
                return True
        except PermissionError as e:
            if i == max_retries - 1:
                print(f"❌ 最终删除失败: {str(e)}")
                return False
            gc.collect()  # 强制垃圾回收
            time.sleep(0.5)  # 等待资源释放
    return False

def connect_to_mysql():
    try:
        conn = mysql.connector.connect(
            host= "localhost",
            user="test2",
            password="123456",
            database="lhwwzx"
        )
        return conn
    except mysql.connector.Error as err:
        print(f"数据库连接失败：{err}")
        return None

def generate_joint_angles_chart(angles_data, analysis_result):
    """生成关节角度图表，返回二进制 PNG 数据"""
    plt.figure(figsize=(15, 10))

    # 绘制每个关节的角度曲线
    for idx, (joint, angles) in enumerate(angles_data.items(), 1):
        plt.subplot(4, 2, idx)
        plt.plot(angles, label='Actual Angle', color='blue', alpha=0.6)

        # 标记理想范围（从分析结果中获取）
        ideal_range = analysis_result[joint].get("ideal_range", (0, 180))
        plt.axhspan(ideal_range[0], ideal_range[1], color='green', alpha=0.2, label='Ideal Range')

        # 标记异常点
        outliers = analysis_result[joint].get("outlier_frames", [])
        plt.scatter(outliers, [angles[i] for i in outliers],
                    color='red', s=20, label='Outliers')

        plt.title(joint)
        plt.xlabel("Frame")
        plt.ylabel("Angle (Degrees)")
        plt.legend()

    plt.tight_layout()

    # 将图表保存到内存中的二进制对象
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()  # 关闭图表释放内存
    binary_data = buf.getvalue()  # 获取二进制数据
    buf.close()

    return binary_data


def save_chart_to_database(conn, chart_name, angles_data, binary_image,video_id):
    """将图表数据和二进制图像存入数据库"""
    cursor = conn.cursor()
    try:
        # 插入图表元数据
        cursor.execute("""
        INSERT INTO chart_metadata 
            (chart_name, related_joints, data_range_start, data_range_end, chart_config,video_id)
         VALUES (%s, %s, %s, %s, %s,%s)
        """, (
            chart_name,
            json.dumps(list(angles_data.keys())),  # 存储关联的关节名
            0,  # 数据起始帧
            max(len(angles) for angles in angles_data.values()) - 1,  # 结束帧
            json.dumps({"style": "seaborn"}),
            video_id  # 图表配置
        ))
        chart_id = cursor.lastrowid

        # 插入二进制图像数据
        cursor.execute("""
        INSERT INTO chart_files 
            (chart_id, image_data, file_format, resolution,video_id)
        VALUES (%s, %s, %s, %s, %s)
        """, (
            chart_id,
            binary_image,  # 二进制 PNG 数据
            'PNG',
            '1920x1080',
            video_id
        ))

        conn.commit()
        print(f"✅ 图表已存储到数据库，ID: {chart_id}")
        return chart_id
    except mysql.connector.Error as err:
        print(f"❌ 存储图表失败: {err}")
        conn.rollback()
        return None
    finally:
        cursor.close()




def save_chart_to_db(conn, chart_data, binary_image,video_id):
    cursor = conn.cursor()
    try:
        # 插入图表元数据
        sql = """
        INSERT INTO chart_metadata (chart_name, related_joints, data_range_start, data_range_end, chart_config,video_id)
        VALUES (%s, %s, %s, %s, %s,%s)
        """
        cursor.execute(sql, (
            chart_data["chart_name"],
            json.dumps(chart_data["related_joints"]),
            chart_data["data_range_start"],
            chart_data["data_range_end"],
            json.dumps(chart_data.get("chart_config",{})),
            video_id
        ))
        chart_id = cursor.lastrowid


        # 插入图表文件信息
        sql = """
        INSERT INTO chart_files (chart_id, image_data, file_format, resolution,video_id)
        VALUES (%s, %s, %s, %s,%s)
        """

        resolution = f"{chart_data['width']}x{chart_data['height']}"
        cursor.execute(sql, (chart_id, binary_image, chart_data["file_format"], f"{chart_data['width']}x{chart_data['height']}",
            video_id))
        conn.commit()
        return chart_id
    except mysql.connector.Error as err:
        print(f"插入图表数据失败: {err}")
        conn.rollback()
        return None
    finally:
        cursor.close()

# 示例数据
angles_data = {
    "Left Elbow": [120, 130, 140],
    "Right Elbow": [125, 135, 145]
}
analysis_result = {
    "Left Elbow": {"ideal_range": (120, 160), "outlier_frames": []},
    "Right Elbow": {"ideal_range": (120, 160), "outlier_frames": []}
}

# 绘制图表
plt.figure(figsize=(15, 10))
for idx, (joint, angles) in enumerate(angles_data.items(), 1):
    plt.subplot(1, 2, idx)
    plt.plot(angles, label='实际角度', color='blue', alpha=0.6)
    ideal_range = analysis_result[joint].get("ideal_range", (0, 180))
    plt.axhspan(ideal_range[0], ideal_range[1], color='green', alpha=0.2, label='理想范围')
    plt.title(joint)
    plt.xlabel("帧数")
    plt.ylabel("角度(度)")
    plt.legend(loc='upper right')
plt.tight_layout()

# 保存文件
output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Analysis_Charts")
os.makedirs(output_dir, exist_ok=True)
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_path = os.path.join(output_dir, f"joint_analysis_{timestamp_str}.png")
plt.savefig(file_path, dpi=150, bbox_inches='tight')
plt.close()
with open(file_path, "rb") as f:
    binary_image = f.read()
# 准备存储数据
chart_data = {
    "chart_name": "关节角度分析",
    "related_joints": list(angles_data.keys()),
    "data_range_start": 0,
    "data_range_end": max(len(v) for v in angles_data.values()) - 1,
    "config": {
        "color_scheme": "seaborn",
        "line_style": "-",
        "marker_size": 5
    },
    "file_format": "PNG",
    "width": 1920,
    "height": 1080
}


def create_tables(conn):
    cursor = conn.cursor()
    try:
        print("⏳ 正在创建表结构...")
        #创建user表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user (
                id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) NOT NULL UNIQUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        # 主视频表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_storage (
                video_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                video_name VARCHAR(255) NOT NULL,
                video_data LONGBLOB NOT NULL,
                upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                file_size BIGINT,
                file_md5 CHAR(32),
                user_id INT UNSIGNED,
                FOREIGN KEY (user_id) REFERENCES user(id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        # 专家资料表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expert_profiles (
                expert_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                video_id INT UNSIGNED NULL,
                name VARCHAR(255) NOT NULL,
                height FLOAT,
                weight FLOAT,
                experience_years INT,
                specialty VARCHAR(255),
                is_global BOOLEAN DEFAULT TRUE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        # 关节角度表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS joint_angles (
                id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                video_id INT UNSIGNED NOT NULL,
                frame INT NOT NULL,
                joint_name VARCHAR(255) NOT NULL,
                angle FLOAT NOT NULL,
                record_time DATETIME NOT NULL,
                FOREIGN KEY (video_id) REFERENCES video_storage(video_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        # 分析结果表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                video_id INT UNSIGNED NOT NULL,
                joint_name VARCHAR(255) NOT NULL,
                ideal_range VARCHAR(255) NOT NULL,
                errors TEXT NOT NULL,
                deviation_percentage FLOAT NOT NULL,
                analysis_time DATETIME NOT NULL,
                FOREIGN KEY (video_id) REFERENCES video_storage(video_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        # JSON 数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS json_data (
                id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                video_id INT UNSIGNED NOT NULL,
                json_content JSON NOT NULL,
                upload_time DATETIME,
                user_id INT UNSIGNED,
                FOREIGN KEY (video_id) REFERENCES video_storage(video_id),
                FOREIGN KEY (user_id) REFERENCES user(id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        # 专家关节数据
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expert_joint_data (
                data_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                expert_id INT UNSIGNED NOT NULL,
                joint_name VARCHAR(255) NOT NULL,
                avg_angle FLOAT,
                min_angle FLOAT,
                max_angle FLOAT,
                FOREIGN KEY (expert_id) REFERENCES expert_profiles(expert_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        # 专家视频表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expert_videos (
                video_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                expert_id INT UNSIGNED NOT NULL,
                video_data LONGBLOB NOT NULL,
                capture_frame INT,
                thumbnail_path VARCHAR(255) NOT NULL,
                FOREIGN KEY (expert_id) REFERENCES expert_profiles(expert_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        # 图表元数据
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chart_metadata (
                chart_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                chart_name VARCHAR(255) NOT NULL,
                related_joints JSON,
                data_range_start INT,
                data_range_end INT,
                chart_config JSON,
                video_id INT UNSIGNED NOT NULL,
                FOREIGN KEY (video_id) REFERENCES video_storage(video_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        # 图表文件表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chart_files (
                file_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                chart_id INT UNSIGNED NOT NULL,
                image_data LONGBLOB NOT NULL,
                file_format VARCHAR(50),
                resolution VARCHAR(50),
                video_id INT UNSIGNED,
                user_id INT UNSIGNED,
                FOREIGN KEY (chart_id) REFERENCES chart_metadata(chart_id),
                FOREIGN KEY (video_id) REFERENCES video_storage(video_id),
                FOREIGN KEY (user_id) REFERENCES user(id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        conn.commit()
        print("✅ 表结构创建成功")

    except mysql.connector.Error as err:
        print(f"❌ 表创建失败: {err}")
        print(f"🧩 出错的 SQL: {cursor.statement}")
        conn.rollback()
    finally:
        cursor.close()



def get_video_analysis(conn, video_id):
        """
        查询指定视频的所有分析数据
        返回格式: {
            'video_info': {...},
            'angles': [...],
            'results': [...],
            'charts': [...]
        }
        """
        cursor = conn.cursor(dictionary=True)  # 使用字典游标
        data = {'video_info': None, 'angles': [], 'results': [], 'charts': []}

        try:
            # 1. 获取视频基本信息
            cursor.execute("""
                SELECT video_id, video_name, upload_time, file_size 
                FROM video_storage 
                WHERE video_id = %s
            """, (video_id,))
            data['video_info'] = cursor.fetchone()

            # 2. 获取关节角度数据（按时间排序）
            cursor.execute("""
                SELECT frame, joint_name, angle, record_time 
                FROM joint_angles 
                WHERE video_id = %s 
                ORDER BY frame
            """, (video_id,))
            data['angles'] = cursor.fetchall()

            # 3. 获取分析结果
            cursor.execute("""
                SELECT joint_name, ideal_range, errors, deviation_percentage, analysis_time 
                FROM analysis_results 
                WHERE video_id = %s
            """, (video_id,))
            data['results'] = cursor.fetchall()

            # 4. 获取关联图表
            cursor.execute("""
                SELECT cm.chart_id, cm.chart_name, cf.image_data 
                FROM chart_metadata cm
                JOIN chart_files cf ON cm.chart_id = cf.chart_id
                WHERE cm.video_id = %s
            """, (video_id,))
            data['charts'] = cursor.fetchall()

            return data
        except mysql.connector.Error as err:
            print(f"查询失败: {err}")
            return None
        finally:
            cursor.close()
def insert_video_data(conn, video_binary, video_name):
        """插入视频并返回video_id"""
        conn = connect_to_mysql()
        cursor = conn.cursor()
        try:
            file_md5 = hashlib.md5(video_binary).hexdigest()
            cursor.execute("""
                INSERT INTO video_storage 
                (video_name, video_data, file_size, file_md5)
                VALUES (%s, %s, %s, %s)
            """, (video_name, video_binary, len(video_binary), file_md5))

            video_id = cursor.lastrowid
            conn.commit()
            return video_id  # 返回新插入的视频ID
        except Exception as e:
            conn.rollback()
            return None
        conn = connect_to_mysql()
        if conn:
            chart_id = save_chart_to_db(conn, chart_data, binary_image, video_id)
            if chart_id:
                print(f"📊 图表已存储到数据库，ID：{chart_id}")
            conn.close()


def read_video_binary(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    with open(video_path,'rb') as f:
        return f.read()
def initialize_expert_data(conn):
    cursor = conn.cursor()
    try:


        # 定义专家数据（示例保持原样）
        experts = [
            {
                "name": "艾菲内普尔",
                "height": "171",
                "weight": "61",
                "experience": 4,
                "specialty": "公路赛",
                "is_glabal":True,
                "joint_data": {
                    "Left Elbow": (120, 160),  # (min_a, max_a)
                    "Right Elbow": (120, 160),
                    "Left Knee": (140, 170),
                    "Right Knee": (140, 170),
                    "Left Shoulder": (80, 120),
                    "Right Shoulder": (80, 120),
                    "Left Hip": (70, 110),
                    "Right Hip": (70, 110)
                },
                "videos": [
                    {
                        "path": r"E:\innovation contest\study\game\med1\fei1.mp4",
                        "thumbnail": "/experts/thumbs/pro_a_side.jpg",
                        "key_frame": 120
                    },
                    {
                        "path": r"E:\innovation contest\study\game\med1\fei2.mp4",
                        "thumbnail": "/experts/thumbs/pro_a_front.jpg",
                        "key_frame": 80
                    }
                ]
            },
            {
                "name": "苏浩钰",
                "height": "171",
                "weight": "61",
                "experience": 2,
                "specialty": "公路赛",
                "is_glabal": True,
                "joint_data": {
                    "Left Elbow": (120, 160),
                    "Right Elbow": (120, 160),
                    "Left Knee": (140, 170),
                    "Right Knee": (140, 170),
                    "Left Shoulder": (80, 120),
                    "Right Shoulder": (80, 120),
                    "Left Hip": (70, 110),
                    "Right Hip": (70, 110)
                },
                "videos": [
                    {
                        "path": r"E:\innovation contest\study\game\med1\su1.mp4",
                        "thumbnail": "/experts/thumbs/pro_a_side.jpg",
                        "key_frame": 120
                    },
                    {
                        "path": r"E:\innovation contest\study\game\med1\su1.mp4",
                        "thumbnail": "/experts/thumbs/pro_a_front.jpg",
                        "key_frame": 80
                    }
                ]
            },
            {
                "name": "波加查",
                "height": "176",
                "weight": "66",
                "experience": 5,
                "specialty": "公路赛",
                "is_glabal": True,
                "joint_data": {
                    "Left Elbow": (120, 160),
                    "Right Elbow": (120, 160),
                    "Left Knee": (140, 170),
                    "Right Knee": (140, 170),
                    "Left Shoulder": (80, 120),
                    "Right Shoulder": (80, 120),
                    "Left Hip": (70, 110),
                    "Right Hip": (70, 110)
                },
                "videos": [
                    {
                        "path": r"E:\innovation contest\study\game\med1\bo1.mp4",
                        "thumbnail": "/experts/thumbs/pro_a_side.jpg",
                        "key_frame": 120
                    },
                    {
                        "path": r"E:\innovation contest\study\game\med1\bo1.mp4",
                        "thumbnail": "/experts/thumbs/pro_a_front.jpg",
                        "key_frame": 80
                    }
                ]
            }
        ]  # 保持你原有的数据定义

        # 插入专家数据
        for expert in experts:
            cursor.execute("""
                INSERT INTO expert_profiles 
                (video_id, name, height, weight, experience_years, specialty, is_global)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                None,
                expert['name'],
                float(expert['height']),
                float(expert['weight']),
                int(expert['experience']),
                expert['specialty'],
                expert.get('is_global',True)
            ))
            expert_id = cursor.lastrowid

            # 插入关节数据（保持不变）
            for joint, (min_a, max_a) in expert['joint_data'].items():
                avg = (min_a + max_a) / 2
                cursor.execute("""
                    INSERT INTO expert_joint_data
                    (expert_id, joint_name, avg_angle, min_angle, max_angle)
                    VALUES (%s, %s, %s, %s, %s)
                """, (expert_id, joint, avg, min_a, max_a))
        conn.commit()
    finally:
        cursor.close()

conn = connect_to_mysql()
if conn:
     create_tables(conn)
     #插入专家视频获取video_id
     initialize_expert_data(conn)
     conn.close()

def enhanced_analyze(user_angles,conn):

    expert_data = {}
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT p.name, j.joint_name, j.avg_angle,j.min_angle,j.max_angle
            FROM expert_joint_data j
            JOIN expert_profiles p ON j.expert_id = p.expert_id
            WHERE p.is_global=TRUE
            """)
        for (name, joint, avg, min_a,max_a) in cursor:
            if name not in expert_data:
                expert_data[name] = {}
            expert_data[name][joint] = {
                'avg':avg,
                'range':(min_a,max_a)
            }
    except mysql.connector.Error as err:
        print(f"专家数据查询失败：{err}")

    #进行对比分析
    analysis = {}
    for joint, angles in user_angles.items():
        user_avg = sum(angles)/len(angles) if angles else 0

        #对比每个专家
        expert_comparison = {}
        for expert_name, data in expert_data.items():
            if joint in data:
                expert_range = data[joint]['range']
                deviation = abs(user_avg - data[joint]['avg'])
                match_percent = max(0, 100 - (deviation / 180 * 100))
                expert_comparison[expert_name] = {
                    'deviation': deviation,
                    'match_percent': round(match_percent,1),
                    'expert_range': expert_range
                }
        #综合所有专家
        analysis[joint] = {
            'user_avg': user_avg,
            'expert_comparison': expert_comparison,
            'suggested_range':(
                min([v[joint]['range'][0]for v in expert_data.values() if joint in v]),
                max([v[joint]['range'][1] for v in expert_data.values() if joint in v])
            )if any(joint in v for v in expert_data.values()) else None
        }
    return analysis

conn = connect_to_mysql()
expert_analysis = enhanced_analyze(angles_over_time,conn)
conn.close()


# 生成增强版报告
def generate_enhanced_report(user_analysis, expert_analysis):
    report = "专业骑行姿势对比报告\n\n"

    for joint, data in expert_analysis.items():
        report += f"关节：{joint}\n"
        report += f"- 您的平均角度：{data['user_avg']:.1f}°\n"

        if data['suggested_range']:
            report += f"- 专业建议范围：{data['suggested_range'][0]}~{data['suggested_range'][1]}°\n"

        for expert, comp in data['expert_comparison'].items():
            report += (
                f"- 与{expert}对比：\n"
                f"  偏离值：{comp['deviation']:.1f}°\n"
                f"  符合度：{comp['match_percent']}%\n"
                f"  该选手范围：{comp['expert_range'][0]}~{comp['expert_range'][1]}°\n"
            )

        # 给出优化建议
        if data['suggested_range']:
            if data['user_avg'] < data['suggested_range'][0]:
                report += "建议：适当增加弯曲角度\n\n"
            elif data['user_avg'] > data['suggested_range'][1]:
                report += "建议：适当减少弯曲角度\n\n"
            else:
                report += "状态良好，保持当前角度范围\n\n"

    return report

#视频二进制转换
def insert_video_data(conn,video_binary,video_name):
        """将视频二进制的数据插入数据库"""
        cursor = conn.cursor()
        try:
            if not isinstance(video_binary,bytes):
                raise ValueError("视频数据必须为二进制格式(bytes)")
            file_md5=hashlib.md5(video_binary).hexdigest()
            file_size=len(video_binary)

            #插入数据
            cursor.execute("""
                INSERT INTO video_storage
                  (video_name,video_data,file_size,file_md5)
                  VALUES(%s,%s,%s,%s)
            """,(video_name,video_binary,file_size,file_md5))
            video_id=cursor.lastrowid
            conn.commit()
            print(f"视频已经存储在数据库，ID:{cursor.lastrowid}")
            return video_id
        except mysql.connector.Error as err:
            print(f"视频储存失败:{err}")
            conn.rollback()
            return None
        finally:
            cursor.close()
#插入关节角度数据
def insert_joint_angles(conn, angles_over_time, timestamp,video_id):
    cursor = conn.cursor()
    try:
       for joint_name, angles in angles_over_time.items():
          for frame, angle in enumerate(angles):
              if angle is None:
                  continue
              cursor.execute("""
                    INSERT INTO joint_angles 
                    (video_id, frame, joint_name, angle, record_time)
                    VALUES (%s, %s, %s, %s, %s)
                """, (video_id, frame, joint_name, angle, timestamp))
       conn.commit()
    except Exception as e:
          print(f"插入关节角度失败:{str(e)}")
          conn.rollback()
    finally:
        cursor.close()
def insert_analysis_results(conn, analysis_result, timestamp,video_id):
    cursor = conn.cursor()
    try:
       for joint_name, data in analysis_result.items():
          cursor.execute("""
                INSERT INTO analysis_results 
                (video_id, joint_name, ideal_range, errors, deviation_percentage, analysis_time)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (video_id, joint_name, str(data['ideal_range']),
                  ', '.join(data['errors']), data['deviation_percentage'], timestamp))
       conn.commit()
    except Exception as e:
        print(f"插入分析结果失败:{str(e)}")
        conn.rollback()
    finally:
        cursor.close()
# 插入 JSON 数据
def insert_json_data(conn, json_str, timestamp, video_id):
    cursor = conn.cursor()
    try:
        sql = """
        INSERT INTO json_data (video_id, json_content, upload_time)
        VALUES (%s, %s, %s)
        """
        cursor.execute(sql, (video_id, json_str, timestamp))
        conn.commit()
        print("JSON 数据插入成功！")
    except mysql.connector.Error as err:
        print(f"插入 JSON 数据失败: {err}")
    finally:
        cursor.close()


# 保留并强化这部分数据库存储功能
def save_analysis_to_db(conn, analysis_data):
    """将分析结果保存到数据库"""
    try:
        cursor = conn.cursor()

        # 1. 保存关节角度数据
        for joint, angles in analysis_data['angles'].items():
            for frame, angle in enumerate(angles):
                cursor.execute(
                    "INSERT INTO joint_angles(frame, joint_name, angle) VALUES (%s, %s, %s)",
                    (frame, joint, angle)
                )

        # 2. 保存分析结果
        for joint, result in analysis_data['results'].items():
            cursor.execute(
                "INSERT INTO analysis_results(joint_name, ideal_range, errors) VALUES (%s, %s, %s)",
                (joint, str(result['ideal_range']), ', '.join(result['errors']))
            )

        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"数据库保存失败: {str(e)}")
        return False

db_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
file_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
def process_video_with_drawing(video_path):
    mp_pose=mp.solutions.pose
    pose=mp_pose.Pose()
    output_path=os.path.splitext(video_path)[0]+"_processed.mp4"
    with VideoCaptureContext(video_path)as cap:
        fps=cap.get(cv2.CAP_PROP_FPS)
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        out=cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        while True:
            ret,frame=cap.read()
            if not ret:
                break
            results=pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0)),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(0,255,255))
                )
            out.write(frame)
        out.release()
    return output_path
def process_video_frames(cap):
    angles_over_time = defaultdict(list)
    cap=None
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            # ...处理帧逻辑...
    finally:
        if cap and cap.isOpened():
            cap.release()
            cv2.destroyAllWindows()
        plt.close('all')  # 关闭所有Matplotlib图形
    return angles_over_time

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

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
    # 新增视频捕获对象
    try:
     cap = cv2.VideoCapture(video_path)
     while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为RGB并处理
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # 计算关节角度
            for start_idx, mid_idx, end_idx, joint_name in JOINT_ANGLES:
                landmarks = results.pose_landmarks.landmark
                a = [landmarks[start_idx].x, landmarks[start_idx].y]
                b = [landmarks[mid_idx].x, landmarks[mid_idx].y]
                c = [landmarks[end_idx].x, landmarks[end_idx].y]
                angle = calculate_angle(a, b, c)
                angles_over_time[joint_name].append(angle)
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()  # 新增释放资源
        cv2.destroyAllWindows()
        plt.close('all')
    return angles_over_time


def process_video_with_cleanup(video_path):
    """带自动资源清理的视频处理
    参数:
        video_path: 视频文件路径
    返回:
        tuple: (angles_over_time, analysis_result, report_content)
    """
    try:
        with VideoCaptureContext(video_path) as cap:
            # 1. 获取关节角度数据
            angles_over_time = process_video_frames(cap)

            # 2. 分析关节角度
            analysis_result = analyze_joint_angles(angles_over_time)

            # 3. 连接数据库获取专家数据
            conn = connect_to_mysql()
            if conn:
                try:
                    # 4. 获取专家对比分析
                    expert_analysis = enhanced_analyze(angles_over_time, conn)

                    # 5. 生成专业报告
                    report_content = generate_enhanced_report(analysis_result, expert_analysis)

                    # 6. 生成可视化图表
                    chart_binary = generate_joint_angles_chart(angles_over_time, analysis_result)

                    return angles_over_time, analysis_result, report_content, chart_binary
                finally:
                    conn.close()
            return None, None, None, None

    finally:
        if os.path.exists(video_path):
            safe_delete(video_path)


def process_video_frames(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

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

    angles_over_time = defaultdict(list)

    try:
        with VideoCaptureContext(video_path) as cap:
            while True:
                ret, frame = cap.read()
                if not ret: break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    for start, mid, end, name in JOINT_ANGLES:
                        a = results.pose_landmarks.landmark[start]
                        b = results.pose_landmarks.landmark[mid]
                        c = results.pose_landmarks.landmark[end]

                        if all([a.visibility > 0.5, b.visibility > 0.5, c.visibility > 0.5]):
                            angle = calculate_angle(
                                [a.x, a.y],
                                [b.x, b.y],
                                [c.x, c.y]
                            )
                            angles_over_time[name].append(angle)
                        else:
                            angles_over_time[name].append(0)  # 明确处理缺失值
    except Exception as e:
        print(f"视频处理异常: {str(e)}")
        raise

    return angles_over_time

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


conn = connect_to_mysql()
if not conn:
    print("无法连接到数据库，程序退出")
    exit()


def analyze_joint_angles(angles_data):
    """本地备用分析函数"""
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
    for joint, angles in angles_data.items():
        # 过滤掉 None 值
        valid_angles = [ang for ang in angles if ang is not None]
        if not valid_angles:
            analysis_result[joint] = {
                "ideal_range": IDEAL_ANGLES.get(joint, (0, 180)),
                "errors": ["没有检测到有效角度数据"],
                "outlier_frames": [],
                "deviation_percentage": 0.0
            }
            continue

        ideal = IDEAL_ANGLES.get(joint, (0, 180))
        outliers = [i for i, ang in enumerate(valid_angles) if not (ideal[0] <= ang <= ideal[1])]
        analysis_result[joint] = {
            "ideal_range": ideal,
            "errors": [f"检测到 {len(outliers)} 帧异常"] if outliers else [],
            "outlier_frames": outliers,
            "deviation_percentage": round(len(outliers) / len(valid_angles) * 100, 2)
        }
    return analysis_result




try:
    video_path = extract_video_data(conn)
    cap = cv2.VideoCapture(video_path)
    print(f"🎉 成功获取视频文件: {video_path}")
    conn = connect_to_mysql()
    if conn:
        processed_video_path =process_video_with_drawing(video_path)
        with open(processed_video_path, 'rb') as f:
            processed_binary=f.read()
        processed_video_id=insert_video_data(
            conn,
            processed_binary,
            f"processed_{os.path.basename(video_path)}"
        )
        if not processed_video_id:
            print("处理视频失败，终止流程")
            exit()

        # 插入视频数据
       # with open(video_path, 'rb') as f:
        #    video_binary = read_video_binary(video_path)
       # video_name = os.path.basename(video_path)
       # video_id = insert_video_data(conn, video_binary, video_name)
       # if not video_id:
        #    print("添加video_id失败")
        # 处理视频获取角度数据
        angles_over_time = process_video_frames(processed_video_path)  # 需要实现这个函数

        # 存储数据
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        insert_joint_angles(conn, angles_over_time,db_timestamp, processed_video_id)
        analysis_result = analyze_joint_angles(angles_over_time)
        insert_analysis_results(conn,analysis_result, timestamp,processed_video_id)

        # 生成并保存图表
        chart_binary = generate_joint_angles_chart(angles_over_time, analysis_result)
        chart_data = {
            "chart_name": "关节角度分析",
            "related_joints": list(angles_over_time.keys()),
            "data_range_start": 0,
            "data_range_end": max(len(v) for v in angles_over_time.values()) - 1,
            "config": {
                "color_scheme": "seaborn",
                "line_style": "-",
                "marker_size": 5
            },
            "file_format": "PNG",
            "width": 1920,
            "height": 1080,
            "video_id": processed_video_id  # 确保包含video_id
        }
        save_chart_to_database(conn, "关节角度分析", angles_over_time, chart_binary,processed_video_id)

        conn.commit()
        conn.close()
    if not os.path.exists(video_path):
        print(f"错误：文件 '{video_path}' 不存在")
        exit()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 '{video_path}'")
        exit()

    # 添加视频参数输出
    print(f"视频信息：")
    print(f"- 格式: {cap.get(cv2.CAP_PROP_FORMAT)}")
    print(f"- 总帧数: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    print(f"- 帧率: {cap.get(cv2.CAP_PROP_FPS)}")

    # 获取视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc=cv2.VideoWriter_fourcc(*'XVID')
    # 定义视频写入对象

    translation_map = {
        "joint_analysis": "关节分析",
        "knee": "膝盖",
        "hip": "髋关节",
        "ankle": "踝关节",
        "angle_changes": "角度变化",
        "ideal_range": "理想范围",
        "current_issues": "现存问题",
        "deviation_percentage": "偏离百分比",
        "training_recommendations": "训练建议",
        "focus_joints": "重点关注关节",
        "optimal_angles": "优化角度建议"
    }


    def translate_keys(data):
        """递归转换字典键名"""
        if isinstance(data, dict):
            return {translation_map.get(k, k): translate_keys(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [translate_keys(item) for item in data]
        return data


    try:
        angles_json = json.dumps(angles_over_time)
        compressed = zlib.compress(angles_json.encode('utf-8'))
        encoded = base64.b64encode(compressed).decode('utf-8')

        url = "https://api.deepseek.com/chat/completions"

        # 定义系统提示词
        system_prompt = """你是一个运动生物力学分析专家。请按以下结构化格式响应,并且用中文回答：
            {
                "joint_analysis": {
                    "关节名称": {
                        "angle_changes": 角度变化描述,
                        "ideal_range": [最小角度, 最大角度],
                        "current_issues": [异常问题描述],
                        "deviation_percentage": 偏离标准值的百分比
                    }
                },
                "training_recommendations": [训练建议],
                "focus_joints": [需要关注的关节列表],
                "optimal_angles": {
                    "关节名称": 推荐角度值
                }
            }"""

        # 使用字典结构构建请求体
        payload = {
            "messages": [
                {
                    "content": system_prompt,
                    "role": "system"
                },
                {
                    "content": f"骑行姿势数据：{encoded}",  # 确保 encoded 变量已定义
                    "role": "user"
                }
            ],
            "model": "deepseek-chat",
            "frequency_penalty": 0,
            "max_tokens": 2048,
            "presence_penalty": 0,
            "response_format": {
                "type": "text"
            },
            "stop": None,
            "stream": False,
            "stream_options": None,
            "temperature": 1,
            "top_p": 1,
            "tools": None,
            "tool_choice": "none",
            "logprobs": False,
            "top_logprobs": None
        }


        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': ''#这里要加上api的密码
        }

        # 使用 json 参数自动序列化
        response = requests.post(url, headers=headers, json=payload)
        print(response.text)
        response_data = response.json()
        content = response_data['choices'][0]['message']['content']
        json_start = content.find('{')
        json_end = content.rfind('}')
        json_str = content[json_start:json_end + 1]
    except Exception as e:
        print(f"保存分析结果时发生错误：{str(e)}")


    def _format_api_response(api_data):
        """
        格式化API响应为统一结构
        """
        formatted = {}
        for joint in api_data.get("joints", []):
            formatted[joint["name"]] = {
                "ideal_range": (
                    joint.get("ideal_min", 0),
                    joint.get("ideal_max", 180)
                ),
                "errors": joint.get("error_messages", []),
                "outlier_frames": joint.get("outlier_frames", [])
            }
        return formatted


    def _get_fallback_result(angles_data):
        """
        API失败时的备用方案
        """
        print("使用本地备用分析规则")
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

        result = {}
        for joint, angles in angles_data.items():
            ideal = IDEAL_ANGLES.get(joint, (0, 180))
            outliers = [i for i, ang in enumerate(angles) if not (ideal[0] <= ang <= ideal[1])]
            result[joint] = {
                "ideal_range": ideal,
                "errors": [f"检测到{len(outliers)}帧异常（本地规则）"] if outliers else [],
                "outlier_frames": outliers
            }
        return result

    # 视频处理完成后，插入关节角度数据
    conn = connect_to_mysql()
    if conn:
        insert_joint_angles(conn, angles_over_time, db_timestamp,processed_video_id)
        expert_analysis = enhanced_analyze(angles_over_time,conn)
        conn.close()

    # 生成分析结果后，插入分析结果
    try:
        analysis_result = analyze_joint_angles(angles_over_time) or _get_fallback_result(angles_over_time)
    except Exception as e:
        print(f"姿势分析失败：{str(e)}")
        analysis_result = _get_fallback_result(angles_over_time)

    conn = connect_to_mysql()
    if conn:
        insert_analysis_results(conn, analysis_result, db_timestamp,processed_video_id)
        conn.close()

    # 生成 JSON 数据后，插入 JSON 数据
    json_str = json.dumps(analysis_result)

    conn = connect_to_mysql()
    if conn:
        insert_json_data(conn, json_str, db_timestamp,processed_video_id)
        conn.close()
    # 保存对比报告
    report_content = generate_enhanced_report(analysis_result, expert_analysis)
    desktop_path = os.path.expanduser("~/Desktop")
    report_path = os.path.join(desktop_path, f"专业对比报告_{file_timestamp}.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"生成专业对比报告：{report_path}")
    # 修改后的可视化函数
    def plot_analysis(angles_data, analysis_result):
        if not analysis_result:
            print("警告：缺少分析结果，跳过可视化")
            return

        plt.figure(figsize=(15, 10))

        for idx, (joint, angles) in enumerate(angles_data.items(), 1):
            try:
                plt.subplot(4, 2, idx)

                # 绘制实际角度
                plt.plot(angles, label='实际角度', color='blue', alpha=0.6)

                # 绘制理想范围
                ideal = analysis_result[joint]["ideal_range"]
                outliers = analysis_result.get(joint, {}).get("outlier_frames", [])
                plt.axhspan(ideal[0], ideal[1], color='green', alpha=0.2, label='理想范围')

                # 标记异常点
                if len(outliers) == len(angles):
                    plt.scatter(outliers, [angles[i] for i in outliers if i < len(angles)],
                                color='red', s=20, label='异常角度')
                # 添加标注
                plt.title(joint)
                plt.xlabel("帧数")
                plt.ylabel("角度(度)")
                plt.legend()
            except (KeyError, IndexError) as e:
                print(f"跳过关节{joint}的可视化:{str(e)}")
                continue
            plt.tight_layout()
            try:
                plt.show()
            except RuntimeError as e:
                print(f"无法显示图表:{str(e)}")


    # 进行姿势分析
    try:
        analysis_result = analyze_joint_angles(angles_over_time) or _get_fallback_result(angles_over_time)
    except Exception as e:
        print(f"姿势分析失败：{str(e)}")
        analysis_result = _get_fallback_result(angles_over_time)
    # 打印分析报告
    print("\n骑行姿势分析报告：")
    if isinstance(analysis_result, dict) and analysis_result:
        for joint, data in angles_over_time.items():
            # 强制类型检查
            data = analysis_result.get(joint, {}) if isinstance(analysis_result, dict) else {}
            print(f"\n[{joint}]")

            # 安全获取数据
            ideal_range = data.get("ideal_range", "未知")
            errors = data.get("errors", [])
            errors = errors if isinstance(errors, list) else []

            print(f"理想角度范围：{ideal_range}")
            if errors:
                print("检测到问题：" + "\n".join(errors))
            else:
                print("角度变化在正常范围内")
    else:
        print("错误：分析结果格式无效")

    # 生成增强型可视化
    try:
        plot_analysis(angles_over_time, analysis_result)
    except Exception as e:
        print(f"可视化处理失败：{str(e)}")


    # 绘制角度变化曲线
    plt.figure()
    for joint_name, angles in angles_over_time.items():
        plt.plot(angles, label=joint_name)
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.title("Joint Angles Over Time")
    plt.legend()

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename1 = os.path.join(desktop_path, f"joint_angles_plot_{file_timestamp}.png")
    plt.savefig(plot_filename1, dpi=300, bbox_inches='tight')
    print(f"关节点角度折线图已经保存到：{plot_filename1}")

    # 显示第一个折线图
    plt.show()


    # 定义plot_analysis 函数
    def plot_analysis(angles_data, analysis_result):
        if not analysis_result:
            print("无法生成可视化：缺少分析结果")
            return None

        plt.figure(figsize=(15, 10))
        for idx, (joint, angles) in enumerate(angles_data.items(), 1):
            plt.subplot(4, 2, idx)
            plt.plot(angles, label='实际角度', color='blue', alpha=0.6)
            ideal = analysis_result.get(joint, {}).get("ideal_range", [0, 180])
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

       #将图表保存为内存中的二进制数据
        buf = BytesIO()
        plt.savefig(buf,format='png',dpi=300,bbox_inches='tight')
        plt.close()
        binary_data = buf.getvalue()
        buf.close()

        return binary_data

    # 调用 plot_analysis 函数
    plot_analysis(angles_over_time, analysis_result)

finally:
    if 'cap' in locals() and hasattr(cap, 'isOpened') and cap.isOpened():
        cap.release()
        print("✅ 视频捕获资源已释放")

    cv2.destroyAllWindows()
    plt.close('all')

    # 关闭数据库连接
    if 'conn' in locals() and hasattr(conn, 'is_connected') and conn.is_connected():
        conn.close()
        print("✅ 数据库连接已关闭")

    # 删除临时文件（带重试机制）
    if 'video_path' in locals() and os.path.exists(video_path):
        max_retries = 3
        for i in range(max_retries):
            try:
                os.remove(video_path)
                print(f"✅ 成功删除临时文件: {video_path}")
                break
            except PermissionError as e:
                if i == max_retries - 1:
                    print(f"❌ 最终删除失败: {str(e)}")
                time.sleep(1)  # 等待1秒再重试
                gc.collect()  # 强制垃圾回收


    # 在已有代码中找到视频处理的主流程部分（通常在视频处理完成后）
    def main_processing_flow(video_path):
        conn = connect_to_mysql()
        if not conn:
            print("数据库连接失败")
            return

        try:
            # 1. 插入视频数据并获取video_id
            with open(video_path, 'rb') as f:
                video_binary = f.read()
            video_id = insert_video_data(conn, video_binary, os.path.basename(video_path))
            initialize_expert_data(conn, video_id)
            if not video_id:
                print("❌ 视频存储失败，终止流程")
                return

            # 2. 处理视频帧获取角度数据（已有代码）
            angles_over_time = process_video_frames(video_path)  # 假设这是你的角度分析函数

            # 3. 插入关节角度数据
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_joint_angles(conn,angles_over_time, datetime.now(),video_id)

            # 4. 执行姿势分析并插入结果
            analysis_result = analyze_joint_angles(angles_over_time)  # 你的分析函数
            insert_analysis_results(conn,analysis_result,datetime.now(),video_id)

            # 5. 生成并保存图表
            chart_binary = generate_joint_angles_chart(angles_over_time, analysis_result)
            save_chart_to_database(conn,"关节角度分析",angles_over_time,chart_binary,video_id)
            #插入json数据
            json_str=json.dumps(analysis_result)
            insert_json_data(conn, json_str,timestamp,video_id)
            conn.commit()
            print(f"✅ 视频 {video_id} 分析数据已完整存储")
            save_chart_to_db(conn, chart_data, chart_binary, video_id)

        except Exception as e:
            conn.rollback()
            print(f"处理失败: {str(e)}")
        finally:
            if conn:conn.close()
