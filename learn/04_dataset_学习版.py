"""
文件：04_dataset_学习版.py
功能：学习数据集加载和处理模块
难度：⭐⭐⭐⭐（中等偏难）
学习目标：
1. 理解类（class）的定义和使用
2. 理解面向对象编程（OOP）的基本概念
3. 理解构造函数（__init__方法）
4. 理解实例变量和方法
5. 理解文件路径操作（os.path.join）
6. 理解列表推导式和循环
7. 理解字典操作和JSON文件处理
8. 理解numpy数组操作
9. 理解4x4变换矩阵

这个文件的作用是：解析不同数据集格式的数据，统一转换为标准格式供后续处理
主要知识点：类、文件I/O、数据转换、矩阵运算
"""

# ============ 知识点1：导入模块 ============
# 这里导入了很多常用的科学计算和图像处理库
import csv              # 处理CSV文件
import glob             # 文件名模式匹配
import os               # 操作系统接口
import cv2              # OpenCV计算机视觉库
import numpy as np      # 数值计算库
import torch            # PyTorch深度学习框架
import trimesh          # 3D网格处理库
from PIL import Image    # Python图像处理库
import json             # JSON文件处理
from pathlib import Path # 路径处理

# 导入项目内部的工具函数
from gaussian_splatting.utils.graphics_utils import focal2fov
from scipy.spatial.transform import Rotation as R

# ============ 知识点2：异常处理导入 ============
# try-except块：尝试导入可选依赖，如果失败就跳过
# pyrealsense2 是Intel RealSense相机SDK，只有在使用RealSense相机时才需要
try:
    import pyrealsense2 as rs
except Exception:
    pass

# ============ 知识点3：注释和文档 ============
# 这是一个多行注释，解释了数据处理的整体思路
# We retain the input interfaces for ground-truth depth and other monocular depth estimations (e.g., from DepthAnything).
# In RGB-only scenarios, the first channel of the RGB image is used as a placeholder for depth input.

## ====================================data parser========================================

# ============ 知识点4：类定义 ============
# dl3dvParser 是一个数据解析器类，专门处理dl3dv数据集格式
class dl3dvParser:
    """
    dl3dv数据集解析器

    学习目标：
    - 理解类的基本结构
    - 理解__init__方法（构造函数）
    - 理解self参数
    - 理解实例变量
    """

    def __init__(self, input_folder, config):
        """
        构造函数：初始化解析器对象

        知识点：构造函数
        - __init__方法在创建对象时自动调用
        - self代表对象本身（当前创建的这个解析器对象）
        - input_folder: 数据集文件夹路径
        - config: 配置字典

        理解self的例子：
        假设我们创建两个解析器对象：
        parser1 = dl3dvParser("/data/scene1", config)
        parser2 = dl3dvParser("/data/scene2", config)

        那么：
        - parser1.input_folder = "/data/scene1"  (parser1对象的input_folder属性)
        - parser2.input_folder = "/data/scene2"  (parser2对象的input_folder属性)

        self就是指向当前对象的指针，让每个对象都能保存自己的数据
        """
        # self.input_folder：给当前对象添加一个名为input_folder的属性
        # 这个属性值就是传入的参数input_folder
        # 每个dl3dvParser对象都会有自己的input_folder属性
        self.input_folder = input_folder
        # ============ 知识点5：字典访问 ============
        # 从配置字典中读取数据集的开始和结束帧索引
        # config["Dataset"]["begin"] 表示：config字典 -> Dataset子字典 -> begin值
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"]

        # ============ 知识点6：文件路径模式匹配 ============
        # glob.glob() 找到所有匹配的文件路径
        # f"{variable}/path" 是字符串格式化语法（f-string）
        # [self.begin:self.end] 是列表切片操作
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]

        # ============ 知识点7：len()函数 ============
        # len() 返回列表的长度（元素个数）
        self.n_img = len(self.color_paths)

        # ============ 知识点8：调用其他方法 ============
        # 在__init__中调用自己的其他方法来完成初始化
        self.load_poses(os.path.join(self.input_folder, "cameras.json"))

    def load_poses(self, pose_file):
        """
        从JSON文件中加载相机位姿

        为什么需要加载位姿？
        =====================================================
        答疑：为什么dataset程序里有加载位姿，但输入参数中没有位姿？

        1. 位姿来源：位姿是从数据集中预先准备的文件中加载的！
           - 不是从函数参数传入的
           - 是从磁盘文件中读取的

        2. 数据集结构：典型的数据集包含：
           - 图像文件（RGB/深度）
           - 位姿文件（相机位置和朝向）
           - 其他元数据

        3. 实际文件位置：
           - dl3dv数据集：input_folder/cameras.json
           - KITTI数据集：input_folder/gt/*.txt

        4. 为什么这么设计？
           - 位姿是"ground truth"（真实值）
           - 用于评估SLAM算法的准确性
           - 模拟真实场景中的相机运动

        =====================================================

        知识点：
        - 文件读取（with open）
        - JSON解析（json.load）
        - 列表操作
        - 循环（for循环）
        - 矩阵运算（numpy）
        - 字典构建
        """
        # ============ 知识点9：文件读取和JSON解析 ============
        # with语句：自动处理文件打开和关闭
        # json.load()：将JSON文件解析为Python字典
        with open(pose_file, "r") as f:
            all_poses = json.load(f)

        # ============ 知识点10：列表切片 ============
        # 从all_poses中选择指定范围的位姿数据
        selected_poses = all_poses[self.begin:self.end]

        # ============ 知识点11：numpy数组操作 ============
        # np.array() 创建numpy数组
        # selected_poses[0]["cam_trans"] 获取第一个位姿的平移向量
        init_trans = np.array(selected_poses[0]["cam_trans"])

        # ============ 知识点12：初始化空列表 ============
        self.poses = []    # 存储逆变换矩阵
        self.frames = []   # 存储帧信息字典

        # ============ 知识点13：for循环和enumerate ============
        # enumerate() 返回索引和值的组合
        # i是索引，pose是位姿数据字典
        for i, pose in enumerate(selected_poses):
            # ============ 知识点14：字典解包 ============
            # 从位姿字典中提取四元数和平移向量
            qx, qy, qz, qw = pose["cam_quat"]  # 四元数
            tx, ty, tz = pose["cam_trans"]      # 平移向量

            # ============ 知识点15：四元数到旋转矩阵转换 ============
            # R.from_quat() 将四元数转换为旋转矩阵
            rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

            # ============ 知识点16：4x4变换矩阵构建 ============
            # np.eye(4) 创建4x4单位矩阵
            # [:3, :3] 表示前3行前3列（旋转部分）
            # [:3, 3] 表示前3行的第4列（平移部分）
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = [tx, ty, tz] - init_trans

            # ============ 知识点17：矩阵求逆 ============
            # np.linalg.inv() 计算矩阵的逆
            # 这里计算世界坐标系到相机坐标系的变换
            inv_pose = np.linalg.inv(transform_matrix)
            self.poses.append(inv_pose)

            # ============ 知识点18：字典构建 ============
            # 为每一帧构建信息字典
            frame = {
                "file_path": self.color_paths[i],           # RGB图像路径
                "depth_path": self.color_paths[i],          # 深度图像路径（这里用RGB代替）
                "mono_depth_path": self.color_paths[i],     # 单目深度路径（这里用RGB代替）
                "transform_matrix": transform_matrix.tolist(), # 变换矩阵（转换为列表）
            }
            self.frames.append(frame)

# ============ 知识点19：另一个数据解析器类 ============
class KITTIParser:
    """
    KITTI数据集解析器（另一个数据集格式的例子）

    学习目标：
    - 理解不同的数据格式处理
    - 理解文件路径模式匹配的复杂用法
    - 理解文本文件读取（np.loadtxt）
    """

    def __init__(self, input_folder, config):
        """KITTI数据集初始化"""
        self.input_folder = input_folder
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"]

        # 文件路径处理（类似dl3dvParser）
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.n_img = len(self.color_paths)

        # ============ 知识点20：复杂路径模式 ============
        # f"{self.input_folder}/gt/*.txt" 匹配gt文件夹下的所有txt文件
        self.load_poses(f"{self.input_folder}/gt/*.txt")

    def load_poses(self, path):
        """从txt文件中加载KITTI位姿"""
        self.poses = []
        self.frames = []

        # ============ 知识点21：复杂文件匹配 ============
        # glob.glob() 找到所有匹配的位姿文件，然后切片选择范围
        pose_files = sorted(glob.glob(path))[self.begin:self.end]

        # ============ 知识点22：文本文件读取 ============
        # np.loadtxt() 从文本文件读取数值数据
        # delimiter=' ' 表示空格分隔
        # reshape(4,4) 重塑为4x4矩阵
        init_trans = np.loadtxt(pose_files[0], delimiter=' ').reshape(4, 4)[:3,3]

        for i in range(self.n_img):
            # 读取每一帧的位姿矩阵
            pose = np.loadtxt(pose_files[i], delimiter=' ').reshape(4, 4)

            # ============ 知识点23：矩阵操作 ============
            # 减去初始位置，实现坐标系归一化
            pose[:3,3] = pose[:3,3] - init_trans

            # 计算逆变换
            inv_pose = np.linalg.inv(pose)
            self.poses.append(inv_pose)

            # 构建帧信息
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": pose.tolist(),
            }
            self.frames.append(frame)

# ============ 知识点24：数据加载函数 ============
def load_dataset(model_params, source_path, config=None):
    """
    数据集加载函数

    学习目标：
    - 理解条件分支（if-elif-else）
    - 理解字符串比较
    - 理解函数返回值
    - 理解模块导入和使用
    """

    # ============ 知识点25：条件判断 ============
    # 根据数据集类型创建对应的解析器
    if config["Dataset"]["type"] == "dl3dv":
        # dl3dv数据集格式
        parser = dl3dvParser(source_path, config)
    elif config["Dataset"]["type"] == "kitti":
        # KITTI数据集格式
        parser = KITTIParser(source_path, config)
    # ... 其他数据集类型 ...

    # ============ 知识点26：返回解析器对象 ============
    return parser

# ============ 练习题 ============
"""
练习1：理解数据流
请回答：
1. dl3dvParser.__init__ 方法做了什么？
2. load_poses 方法为什么需要计算矩阵的逆？
3. transform_matrix.tolist() 的作用是什么？

练习2：修改代码
请修改 dl3dvParser 类，让它能够处理实际的深度图像文件
（而不是用RGB图像代替）

练习3：扩展功能
请为这个模块添加一个新的数据集解析器类，支持自定义的数据格式

扩展阅读：
=========
如果你对"为什么需要加载位姿"还有疑问，请查看：
07_位姿概念详解.py - 专门解释位姿的概念和数据流
"""

if __name__ == "__main__":
    # ============ 知识点27：主函数执行 ============
    # 当直接运行这个文件时，执行的代码
    print("这是一个数据集处理模块的学习版本")
    print("主要知识点：类定义、文件I/O、矩阵运算、数据解析")
