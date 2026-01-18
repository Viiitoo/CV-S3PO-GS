"""
文件：02_time_utils_学习版.py
功能：学习时间工具模块
难度：⭐⭐（简单）
学习目标：
1. 理解类型注解（type hints）
2. 理解可选参数（Optional）
3. 理解类型联合（Union）
4. 理解函数文档字符串（docstring）
5. 理解条件分支（if-else）
6. 理解最大值函数（max）
7. 理解动态属性设置（setattr）
8. 理解关键字参数（*args前面的*）

这个文件的作用是：处理视频帧的时间归一化，将帧索引转换为0-1之间的时间值
"""

# ============ 知识点1：从未来版本导入 ============
# Python的类型注解功能，让代码更清晰，但不影响运行
from __future__ import annotations

# ============ 知识点2：导入类型注解工具 ============
# typing 模块提供了类型注解相关的工具
from dataclasses import dataclass
from typing import Optional, Union

# ============ 知识点3：类型别名 ============
# Number 是一个类型别名，表示可以是 int（整数）或 float（浮点数）
# Union[int, float] 表示"整数或浮点数"
Number = Union[int, float]


def normalize_frame_time(frame_idx: Number,
                         num_frames: Optional[int] = None,
                         *,
                         clamp: bool = True,
                         eps: float = 1e-8) -> float:
    """
    将帧索引归一化到0-1的时间范围（EH风格）
    
    知识点：函数文档字符串（docstring）
    - 用三引号 """ 包围的字符串
    - 放在函数定义的第一行，用于说明函数的作用
    - 可以用来说明参数、返回值、使用示例等
    
    参数说明：
        frame_idx: 帧索引（可以是整数或浮点数）
                   例如：第0帧、第1帧、...、第99帧
        
        num_frames: 总帧数（可选，整数类型）
                    Optional[int] = None 表示：
                    - 可以是整数类型（int）
                    - 也可以是 None（没有值）
                    - 默认值是 None
        
        *, clamp, eps: 关键字参数
                       * 符号后面的参数必须用"参数名=值"的方式传入
                       不能直接传位置参数
        
        clamp: 是否限制结果在0-1范围内（默认True）
        
        eps: 很小的正数，防止除以0（默认1e-8，即0.00000001）
    
    返回值：
        float: 归一化后的时间值，范围通常在[0, 1]
    
    计算公式：
        如果有总帧数：t = frame_idx / (num_frames - 1)
        例如：总共100帧（索引0-99），第50帧对应的时间 = 50 / 99 ≈ 0.505
    
    示例：
        normalize_frame_time(50, num_frames=100)  # 返回 0.505...
        normalize_frame_time(0, num_frames=100)   # 返回 0.0
        normalize_frame_time(99, num_frames=100)  # 返回 1.0
    """
    # ============ 知识点4：类型转换 ============
    # float() 函数将参数转换为浮点数
    # 确保后续计算都是浮点数运算，避免整数除法问题
    idx = float(frame_idx)

    # ============ 知识点5：条件判断分支 ============
    # if-else 用于条件分支
    # num_frames is None 检查是否没有提供总帧数
    if num_frames is None:
        # 如果没有提供总帧数，直接使用帧索引作为时间
        # 这是一个fallback（备选方案），不推荐但可以避免程序崩溃
        t = idx
    else:
        # ============ 知识点6：max函数和数学运算 ============
        # max(eps, float(num_frames - 1)) 取两个值中的较大者
        # 这样即使 num_frames - 1 = 0，也会使用 eps，避免除以0
        # float(num_frames - 1) 将结果转换为浮点数
        denom = max(eps, float(num_frames - 1))
        
        # 计算归一化时间：当前帧索引 / (总帧数 - 1)
        # 例如：第50帧，总共100帧 → 50 / 99 ≈ 0.505
        t = idx / denom

    # ============ 知识点7：条件限制（clamp） ============
    # 如果 clamp=True，确保结果在[0, 1]范围内
    if clamp:
        if t < 0.0: 
            t = 0.0  # 如果小于0，设为0
        if t > 1.0: 
            t = 1.0  # 如果大于1，设为1
    
    # 返回结果，确保是浮点数类型
    return float(t)


def attach_time_to_viewpoint(viewpoint_camera,
                             frame_idx: Number,
                             num_frames: Optional[int] = None,
                             *,
                             attr: str = "time") -> float:
    """
    计算时间值并附加到相机视角对象上
    
    这个函数做了两件事：
    1. 计算归一化时间（调用上面的函数）
    2. 把时间值设置到 viewpoint_camera 对象的属性上
    
    参数：
        viewpoint_camera: 相机视角对象（可以是任意对象，只要有属性就可以）
        frame_idx: 帧索引
        num_frames: 总帧数（可选）
        attr: 要设置的属性名（默认是"time"）
              *, attr="time" 表示这是关键字参数，必须用 attr="..." 的方式传入
    
    返回值：
        float: 计算出的时间值
    
    示例：
        # 假设有一个相机对象 camera
        t = attach_time_to_viewpoint(camera, 50, num_frames=100)
        # 此时 camera.time = 0.505...（约0.505）
        print(camera.time)  # 输出：0.505...
    """
    # 调用上面定义的函数，计算归一化时间
    t = normalize_frame_time(frame_idx, num_frames)
    
    # ============ 知识点8：动态设置对象属性 ============
    # setattr(object, name, value) 动态给对象设置属性
    # 等价于：viewpoint_camera.attr = t
    # 但使用变量名作为属性名更灵活
    
    # 例如：如果 attr="time"
    # setattr(viewpoint_camera, "time", t) 相当于 viewpoint_camera.time = t
    # 如果 attr="normalized_time"
    # setattr(viewpoint_camera, "normalized_time", t) 相当于 viewpoint_camera.normalized_time = t
    setattr(viewpoint_camera, attr, t)
    
    return t


# ============ 测试示例 ============
if __name__ == "__main__":
    print("=== 测试时间归一化功能 ===")
    
    # 测试1：基本功能
    t1 = normalize_frame_time(0, num_frames=100)
    print(f"第0帧（共100帧）的时间：{t1}")  # 应该是 0.0
    
    t2 = normalize_frame_time(50, num_frames=100)
    print(f"第50帧（共100帧）的时间：{t2}")  # 应该是 50/99 ≈ 0.505
    
    t3 = normalize_frame_time(99, num_frames=100)
    print(f"第99帧（共100帧）的时间：{t3}")  # 应该是 1.0
    
    # 测试2：边界情况
    t4 = normalize_frame_time(-10, num_frames=100, clamp=True)
    print(f"负数帧索引（clamp=True）：{t4}")  # 应该是 0.0（被限制）
    
    # 测试3：附加到对象
    class Camera:
        pass
    
    camera = Camera()
    attach_time_to_viewpoint(camera, 25, num_frames=100)
    print(f"相机对象的时间属性：{camera.time}")  # 应该是 25/99 ≈ 0.253
