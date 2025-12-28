#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从viz文件夹中的图片提取左上角子图（Ground Truth）和右上角子图（Rendered rgb），
将两者并排放置并组合成视频
"""

import os
import glob
import re
from PIL import Image
import numpy as np

# 尝试导入视频处理库
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False
        print("警告：未安装imageio，将使用opencv。如果opencv也未安装，请安装其中之一：pip install imageio 或 pip install opencv-python")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

if not HAS_IMAGEIO and not HAS_CV2:
    raise ImportError("请安装imageio或opencv-python：pip install imageio[ffmpeg] 或 pip install opencv-python")


def extract_number(filename):
    """从文件名中提取数字用于排序"""
    match = re.search(r'(\d+)\.png', filename)
    return int(match.group(1)) if match else 0


def extract_top_subplots(image_path):
    """
    提取图片上方的两个子图并拼接
    假设图片使用2x2 subplot布局：
    - 左上：Ground Truth
    - 右上：Rendered rgb
    - 左下：Depth Map
    - 右下：Residual
    
    返回：水平拼接的图片（Ground Truth在左，Rendered rgb在右）
    """
    img = Image.open(image_path)
    width, height = img.size
    
    # 左上角子图（Ground Truth）：左半部分的上半部分
    gt_left = 0
    gt_top = 0
    gt_right = width // 2
    gt_bottom = height // 2
    
    # 右上角子图（Rendered rgb）：右半部分的上半部分
    rgb_left = width // 2
    rgb_top = 0
    rgb_right = width
    rgb_bottom = height // 2
    
    # 提取两个子图
    gt_subplot = img.crop((gt_left, gt_top, gt_right, gt_bottom))
    rgb_subplot = img.crop((rgb_left, rgb_top, rgb_right, rgb_bottom))
    
    # 转换为numpy数组
    gt_array = np.array(gt_subplot)
    rgb_array = np.array(rgb_subplot)
    
    # 水平拼接（Ground Truth在左，Rendered rgb在右）
    combined = np.concatenate([gt_array, rgb_array], axis=1)
    
    return combined


def create_video_with_opencv(frames, output_path, width, height, fps):
    """使用OpenCV创建视频"""
    if not HAS_CV2:
        return False
    
    # 尝试多种编码器
    fourcc_options = [
        ('avc1', 'H.264 (avc1)'),
        ('X264', 'H.264 (X264)'),
        ('mp4v', 'MPEG-4'),
        ('XVID', 'Xvid'),
    ]
    
    video_writer = None
    used_codec = None
    
    for codec, name in fourcc_options:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if video_writer.isOpened():
            print(f"使用编码器: {name}")
            used_codec = codec
            break
        video_writer.release()
    
    if video_writer is None or not video_writer.isOpened():
        print("错误：无法创建OpenCV视频写入器")
        return False
    
    # 写入帧
    for i, frame in enumerate(frames):
        # 确保frame是uint8格式
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 255)).astype(np.uint8)
        
        # PIL打开的是RGB，需要转换为BGR供OpenCV使用
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        
        video_writer.write(frame_bgr)
    
    video_writer.release()
    
    # 检查文件大小
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size < 1024:  # 小于1KB说明写入失败
            print(f"警告：视频文件太小（{file_size}字节），OpenCV写入可能失败")
            try:
                os.remove(output_path)
            except:
                pass
            return False
        else:
            print(f"OpenCV视频创建成功: {output_path} (大小: {file_size / 1024 / 1024:.2f} MB)")
            return True
    
    return False


def create_video_with_imageio(frames, output_path, fps):
    """使用imageio创建视频"""
    if not HAS_IMAGEIO:
        return False
    
    try:
        # 确保frames是uint8格式
        frames_uint8 = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 255)).astype(np.uint8)
            frames_uint8.append(frame)
        
        # 尝试使用libx264编码
        imageio.mimwrite(output_path, frames_uint8, fps=fps, codec='libx264', quality=8)
        file_size = os.path.getsize(output_path)
        print(f"imageio视频创建成功: {output_path} (大小: {file_size / 1024 / 1024:.2f} MB)")
        return True
    except TypeError:
        # 如果quality参数不支持，使用默认参数
        try:
            frames_uint8 = []
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 255)).astype(np.uint8)
                frames_uint8.append(frame)
            
            imageio.mimwrite(output_path, frames_uint8, fps=fps, codec='libx264')
            file_size = os.path.getsize(output_path)
            print(f"imageio视频创建成功: {output_path} (大小: {file_size / 1024 / 1024:.2f} MB)")
            return True
        except ValueError as e2:
            print(f"错误：imageio无法创建视频 - {e2}")
            print("提示：请安装ffmpeg后端：pip install imageio[ffmpeg]")
            return False
    except ValueError as e:
        print(f"错误：imageio无法创建视频 - {e}")
        print("提示：请安装ffmpeg后端：pip install imageio[ffmpeg]")
        return False


def create_video_from_viz(viz_dir, output_video_path, fps=10):
    """
    从viz文件夹中提取所有图片的左上角（Ground Truth）和右上角（Rendered rgb）子图，
    将两者并排拼接后创建视频
    
    Args:
        viz_dir: viz文件夹路径
        output_video_path: 输出视频文件路径
        fps: 视频帧率
    """
    # 获取所有png文件
    image_pattern = os.path.join(viz_dir, "*.png")
    image_files = glob.glob(image_pattern)
    
    if not image_files:
        print(f"警告：在 {viz_dir} 中没有找到png图片")
        return
    
    # 按照文件名中的数字排序
    image_files.sort(key=extract_number)
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 读取第一张图片以确定视频尺寸
    first_combined = extract_top_subplots(image_files[0])
    height, width = first_combined.shape[:2]
    
    print(f"组合图尺寸: {width}x{height} (Ground Truth + Rendered rgb)")
    print(f"开始创建视频: {output_video_path}")
    
    # 收集所有组合图
    frames = []
    
    # 处理每一张图片
    for i, image_path in enumerate(image_files):
        try:
            combined = extract_top_subplots(image_path)
            frames.append(combined)
            
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(image_files)} 张图片")
        except Exception as e:
            print(f"处理 {image_path} 时出错: {e}")
            continue
    
    print(f"开始写入视频...")
    
    # 优先尝试使用OpenCV，如果失败则使用imageio
    success = False
    
    if HAS_CV2:
        print("尝试使用OpenCV创建视频...")
        success = create_video_with_opencv(frames, output_video_path, width, height, fps)
    
    if not success and HAS_IMAGEIO:
        print("尝试使用imageio创建视频...")
        success = create_video_with_imageio(frames, output_video_path, fps)
    
    if not success:
        print("错误：无法创建视频")
        print("请确保安装了以下之一：")
        print("  - opencv-python: pip install opencv-python")
        print("  - imageio[ffmpeg]: pip install imageio[ffmpeg]")
        return
    
    print(f"视频创建完成: {output_video_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从viz文件夹提取左上角(Ground Truth)和右上角(Rendered rgb)子图，并排拼接后创建视频")
    parser.add_argument(
        "--viz_dir",
        type=str,
        default="results/stereo_seq_easy_stereo_seq_easy/2025-12-28-22-41-58/viz",
        help="viz文件夹路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gt_and_rendered_rgb_video.mp4",
        help="输出视频文件路径"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="视频帧率（默认：10）"
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    viz_dir = os.path.abspath(args.viz_dir)
    output_path = os.path.abspath(args.output)
    
    if not os.path.exists(viz_dir):
        print(f"错误：viz文件夹不存在: {viz_dir}")
        exit(1)
    
    create_video_from_viz(viz_dir, output_path, args.fps)
