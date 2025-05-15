# extract_video.py
import mysql.connector
import hashlib
import uuid
import os

import cv2
from pathlib import Path


def connect_to_mysql():
    """创建安全的数据库连接"""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123456",
            database="lhwwzx",
            charset='utf8mb4'
        )
        print("✅ 数据库连接成功")
        return conn
    except mysql.connector.Error as err:
        print(f"❌ 数据库连接失败: {err}")
        return None


def extract_video_data(conn):
    cursor = conn.cursor()
    temp_path = None  # 用于finally清理

    try:
        # 强化查询条件：确保字段存在
        cursor.execute("""
            SELECT video_data, create_time 
            FROM video 
            WHERE video_data IS NOT NULL 
            ORDER BY create_time DESC 
            LIMIT 1
        """)

        if not (result := cursor.fetchone()):
            print("⚠️ 数据库中没有有效视频记录")
            return None

        video_blob, create_time = result

        # 基础数据验证
        if not isinstance(video_blob, bytes) or len(video_blob) < 1024:  # 至少1KB
            print(f"❌ 视频数据异常：类型={type(video_blob)} 大小={len(video_blob)}字节")
            return None

        # 创建带时间戳的临时目录
        temp_dir = os.path.join(
            os.path.expanduser("~/video_exports"),
            create_time.strftime("%Y%m%d")
        )
        os.makedirs(temp_dir, exist_ok=True, mode=0o755)  # 更安全的权限

        # 生成唯一文件名
        temp_filename = f"video_{create_time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}.mp4"
        temp_path = os.path.join(temp_dir, temp_filename)

        # 原子化写入操作
        try:
            with open(temp_path, 'wb') as f:
                f.write(video_blob)
            # 基础文件验证
            if os.path.getsize(temp_path) != len(video_blob):
                raise IOError("文件大小不一致")
        except IOError as e:
            print(f"❌ 文件写入失败: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None

        print(f"✅ 视频已安全导出到: {temp_path}")
        return temp_path

    except Exception as e:
        print(f"❌ 视频提取失败: {str(e)}")
        # 清理已创建的文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return None
    finally:
        cursor.close()
if __name__ == "__main__":
    conn = connect_to_mysql()
    if conn:
        try:
            video_path = extract_video_data(conn)
            print(f"🎉 视频转换成功，最终文件: {video_path}")
        except Exception as e:
            print(f"🔥 处理失败: {str(e)}")
        finally:
            conn.close()