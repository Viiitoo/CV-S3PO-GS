# -*- coding: utf-8 -*-
"""
文件：07_位姿概念详解.py
功能：解释SLAM中位姿（pose）的概念和数据流
难度：⭐⭐⭐（中等重要）
学习目标：
1. 理解什么是位姿
2. 理解位姿在SLAM中的作用
3. 理解位姿数据的来源和格式
4. 理解为什么需要加载位姿文件

这个文件的作用是：解答"为什么dataset程序里有加载位姿，但输入参数中没有位姿"的疑问
"""

# ============ 知识点1：什么是位姿（Pose）？ ============
"""
位姿 = 位置（Position）+ 姿态（Orientation）

在3D空间中，任何一个物体（比如相机）的位姿由6个自由度确定：
- 位置：x, y, z 坐标（3个自由度）
- 姿态：旋转角度（3个自由度，通常用四元数或旋转矩阵表示）

位姿告诉我们：
- 相机在哪里（位置）
- 相机朝向哪里（姿态）
"""

# ============ 知识点2：位姿在SLAM中的作用 ============
"""
为什么SLAM需要位姿？

1. Ground Truth（真实值）：
   - 用于评估SLAM算法的准确性
   - 计算ATE（绝对轨迹误差）

2. 初始化：
   - 为SLAM算法提供初始相机位置
   - 帮助算法快速收敛

3. 仿真和测试：
   - 在数据集上测试算法性能
   - 比较不同算法的效果

4. 真实应用中的替代：
   - 实际应用中没有预知位姿
   - SLAM算法需要自己估计位姿
   - 这里用数据集中的位姿来模拟真实场景
"""

# ============ 知识点3：位姿数据的来源 ============
"""
位姿数据从哪里来？

1. 数据集收集阶段：
   - 使用高精度设备（如激光雷达、GPS）记录相机位姿
   - 专业的数据采集车或无人机
   - 实验室中的运动捕捉系统

2. 后处理计算：
   - 从视频序列中重建相机轨迹（SfM: Structure from Motion）
   - 使用IMU（惯性测量单元）数据
   - 多相机标定技术

3. 合成数据集：
   - 使用计算机图形学生成虚拟场景
   - 模拟相机运动轨迹
"""

def explain_pose_data_flow():
    """解释位姿数据流"""

    print("=== 位姿数据流解释 ===")
    print()

    print("1. 数据集准备阶段（离线）：")
    print("   研究人员/工程师 ──► 采集设备 ──► 原始数据")
    print("                         │")
    print("                         └─► 位姿文件（cameras.json 或 *.txt）")
    print()

    print("2. SLAM训练/测试阶段（在线）：")
    print("   数据集文件夹 ──► DatasetParser ──► 加载位姿")
    print("     │                    │")
    print("     ├─ images/          ├─ self.poses[]")
    print("     └─ cameras.json      └─ self.frames[]")
    print()

    print("3. 实际应用阶段：")
    print("   实时相机流 ──► SLAM算法 ──► 实时位姿估计")
    print("                         │")
    print("                         └─► 没有预知位姿，需要自己计算")
    print()

# ============ 知识点4：位姿数据格式 ============
"""
位姿数据的常见格式：

1. dl3dv格式（JSON）：
{
  "cam_quat": [0.0, 0.0, 0.0, 1.0],  // 四元数 (qx, qy, qz, qw)
  "cam_trans": [0.0, 0.0, 0.0]         // 平移向量 (tx, ty, tz)
}

2. KITTI格式（文本）：
第一行：3x4变换矩阵
[[r11, r12, r13, tx],
 [r21, r22, r23, ty],
 [r31, r32, r33, tz]]

3. 变换矩阵格式（4x4）：
[[r11, r12, r13, tx],
 [r21, r22, r23, ty],
 [r31, r32, r33, tz],
 [0,   0,   0,   1 ]]
"""

def show_pose_formats():
    """展示不同位姿格式"""

    print("=== 位姿数据格式示例 ===")
    print()

    print("1. dl3dv JSON格式：")
    dl3dv_pose = {
        "cam_quat": [0.0, 0.0, 0.0, 1.0],  # 单位四元数（无旋转）
        "cam_trans": [1.0, 2.0, 3.0]        # 位置在(1,2,3)
    }
    print("JSON示例：", dl3dv_pose)
    print()

    print("2. 变换矩阵格式（4x4）：")
    # 创建一个简单的变换矩阵（用列表表示）
    transform_matrix = [
        [1.0, 0.0, 0.0, 1.0],  # x = 1
        [0.0, 1.0, 0.0, 2.0],  # y = 2
        [0.0, 0.0, 1.0, 3.0],  # z = 3
        [0.0, 0.0, 0.0, 1.0]   # 齐次坐标
    ]
    print("4x4变换矩阵：")
    for row in transform_matrix:
        print(row)
    print()

# ============ 知识点5：位姿转换过程 ============
"""
位姿数据处理流程：

原始位姿数据 ──► 解析 ──► 坐标系转换 ──► 存储

1. 解析：从文件格式转换为Python数据结构
2. 坐标系转换：确保坐标系一致性
3. 存储：保存到self.poses和self.frames中
"""

def demonstrate_pose_processing():
    """演示位姿处理过程"""

    print("=== 位姿处理过程演示 ===")
    print()

    # 模拟从JSON文件读取的位姿数据
    raw_pose_data = [
        {"cam_quat": [0.0, 0.0, 0.0, 1.0], "cam_trans": [0.0, 0.0, 0.0]},
        {"cam_quat": [0.0, 0.0, 0.0, 1.0], "cam_trans": [1.0, 0.0, 0.0]},
        {"cam_quat": [0.0, 0.0, 0.0, 1.0], "cam_trans": [2.0, 0.0, 0.0]},
    ]

    print("1. 原始JSON数据：")
    for i, pose in enumerate(raw_pose_data):
        print("   帧" + str(i) + ": 位置" + str(pose['cam_trans']) + ", 旋转" + str(pose['cam_quat']))
    print()

    # 模拟处理过程
    processed_poses = []
    processed_frames = []

    for i, pose in enumerate(raw_pose_data):
        # 步骤1：提取数据
        qx, qy, qz, qw = pose["cam_quat"]
        tx, ty, tz = pose["cam_trans"]

        # 步骤2：构建变换矩阵（这里简化了）
        transform_matrix = [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0]
        ]

        # 步骤3：计算逆变换（这里简化，实际代码中会用numpy.linalg.inv）
        # 逆变换就是取负的平移向量
        inv_pose = [
            [1.0, 0.0, 0.0, -tx],
            [0.0, 1.0, 0.0, -ty],
            [0.0, 0.0, 1.0, -tz],
            [0.0, 0.0, 0.0, 1.0]
        ]

        # 步骤4：存储结果
        processed_poses.append(inv_pose)
        processed_frames.append({
            "file_path": "rgb/frame_" + str(i).zfill(4) + ".png",
            "transform_matrix": transform_matrix  # 已经是list格式
        })

    print("2. 处理后的数据：")
    print("   self.poses: " + str(len(processed_poses)) + "个变换矩阵")
    print("   self.frames: " + str(len(processed_frames)) + "个帧信息")
    print()

# ============ 练习题 ============
"""
练习1：理解位姿概念
请回答：
1. 位姿包含哪些信息？
2. 为什么SLAM需要位姿数据？
3. 位姿数据在实际应用和数据集测试中有何区别？

练习2：理解数据流
请回答：
1. dl3dvParser如何获得位姿数据？
2. KITTIParser的位姿文件格式是什么？
3. 为什么需要计算矩阵的逆（np.linalg.inv）？

练习3：设计数据集
请思考：如果你要创建一个新的SLAM数据集，需要准备哪些文件？
"""

# ============ 主程序 ============
if __name__ == "__main__":
    print("=== SLAM中位姿概念详解 ===")
    print()

    explain_pose_data_flow()
    print("="*50)
    print()

    show_pose_formats()
    print("="*50)
    print()

    demonstrate_pose_processing()
    print("="*50)
    print()

    print("=== 总结：为什么dataset程序里有加载位姿？ ===")
    print("1. 位姿是数据集的一部分，不是输入参数")
    print("2. 用于评估SLAM算法性能（ground truth）")
    print("3. 模拟真实场景中的相机运动轨迹")
    print("4. 文件位置：input_folder/cameras.json 或 gt/*.txt")
    print()
    print("现在你明白为什么需要加载位姿了吧？😊")
