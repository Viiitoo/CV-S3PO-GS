"""
文件：05_slam_main_学习版.py
功能：学习SLAM系统主程序入口
难度：⭐⭐⭐⭐⭐（难，需要理解系统架构）
学习目标：
1. 理解多进程编程（multiprocessing）
2. 理解生产者-消费者模式（队列通信）
3. 理解系统架构设计（前后端分离）
4. 理解CUDA事件和性能测量
5. 理解配置参数的使用
6. 理解SLAM系统的整体工作流程
7. 理解GUI和可视化集成

这个文件的作用是：协调SLAM系统的各个组件，实现实时3D重建
主要知识点：多进程架构、前后端分离、队列通信、系统集成
"""

# ============ 知识点1：标准库导入 ============
# 【语法：import语句】import module_name - 导入整个模块
# 【语法说明】导入Python标准库，可以直接使用模块中的函数和类
import os              # 操作系统接口（文件和目录操作）
import glob            # 文件名模式匹配（通配符查找文件）
import sys             # 系统相关功能（命令行参数、退出等）
import time            # 时间测量（sleep、time等函数）
# 【语法：import as】import module as alias - 导入模块并给别名
# 【语法说明】as np是别名，使用np代替numpy，简化代码
import numpy as np     # 数值计算库，别名为np

# 【语法：from import】from module import name - 从模块导入特定名称
# 【语法说明】只导入需要的类或函数，不需要写module.ClassName
from argparse import ArgumentParser  # 从argparse模块导入ArgumentParser类
from datetime import datetime        # 从datetime模块导入datetime类
import pandas as pd    # 数据处理库，别名为pd（虽然在这个文件中没怎么用到）

# ============ 知识点2：深度学习和科学计算库 ============
# 【语法：import模块】import torch - 导入PyTorch深度学习框架
import torch           # PyTorch深度学习框架（张量操作、神经网络等）

# 【语法：子模块导入】import module.submodule as alias - 导入子模块并给别名
# 【语法说明】torch.multiprocessing是torch的子模块，as mp给别名
import torch.multiprocessing as mp  # PyTorch多进程支持（进程、队列等）

import yaml            # YAML配置文件处理（读取YAML格式的配置文件）
# 【语法：from import】from package import function - 从包中导入函数
# 【语法说明】munch是包名，munchify是其中的函数
from munch import munchify  # 配置字典增强（将字典转换为支持点操作的对象）

# ============ 知识点3：第三方库 ============
import wandb           # Weights & Biases实验跟踪

# ============ 知识点4：项目内部模块导入 ============
# 核心组件
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p

# GUI相关
from gui import gui_utils, slam_gui

# 工具模块
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue

# SLAM核心模块
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd

# 深度估计模型
from mast3r.model import AsymmetricMASt3R

# ============ 知识点5：主SLAM类定义 ============
# 【语法：类定义】class ClassName: - 定义一个新类
# 【语法说明】class是关键字，SLAM是类名，冒号后是类体
#           类名通常使用驼峰命名法（每个单词首字母大写）
class SLAM:
    """
    SLAM系统主类

    学习目标：
    - 理解系统架构设计
    - 理解前后端分离模式
    - 理解多进程通信
    - 理解配置驱动的系统设计

    SLAM系统架构：
    - Frontend（前段）：处理实时数据输入、特征提取、初步跟踪
    - Backend（后段）：优化高斯模型、进行全局优化
    - GUI：可视化界面
    - Queues：进程间通信队列
    """

    # 【语法：方法定义】def method_name(self, param1, param2=default): - 定义类方法
    # 【语法说明】def是关键字，__init__是特殊方法名（构造函数）
    #           self是第一个参数（必须是self），代表对象实例本身
    #           param1是位置参数，param2=default是带默认值的关键字参数
    def __init__(self, config, mast3r_model, save_dir=None):
        """
        SLAM系统初始化

        参数说明：
        - self: 对象实例本身（所有实例方法的第一个参数）
        - config: 配置字典（必需参数）
        - mast3r_model: MASt3R深度估计模型对象（必需参数）
        - save_dir: 保存目录路径（可选参数，默认值为None）

        知识点：CUDA事件计时
        - torch.cuda.Event() 创建CUDA事件用于GPU计时
        - enable_timing=True 启用计时功能
        """
        # ============ 知识点6：GPU性能计时 ============
        # 【语法1：类实例化】torch.cuda.Event() - 调用类的构造函数创建对象
        # 【语法说明】ClassName(arg1=value1, arg2=value2) - 创建类的实例
        # 【语法2：关键字参数】enable_timing=True - 使用关键字参数传递值
        # 【语法说明】func(参数名=值) - 明确指定参数名称，提高代码可读性
        # 【语法3：属性访问】torch.cuda.Event - 通过点操作访问模块的属性（类）
        # 【语法说明】module.submodule.ClassName - 访问嵌套模块中的类
        start = torch.cuda.Event(enable_timing=True)  # 创建开始计时事件对象
        end = torch.cuda.Event(enable_timing=True)    # 创建结束计时事件对象
        # 【语法4：方法调用】object.method() - 调用对象的方法
        # 【语法说明】instance.method() - 在对象实例上调用方法
        start.record()  # 开始计时（record是Event对象的方法）

        # ============ 知识点7：保存基本参数 ============
        # 【语法：实例变量赋值】self.xxx = value - 将值赋给实例变量
        # 【语法说明】self是类方法的第一个参数，代表当前对象实例
        #           通过self.xxx可以在类的任何方法中访问这个变量
        self.config = config          # 保存系统配置字典到实例变量
        self.save_dir = save_dir      # 保存目录路径到实例变量

        # ============ 知识点8：配置参数解析 ============
        # 【语法1：函数调用】munchify() 将普通字典转换为支持点操作的字典对象
        # 【语法说明】函数名(参数) - 调用munch库的munchify函数
        # 【语法2：字典访问】config["model_params"] - 使用方括号访问字典的键
        # 【语法说明】dict["key"] - 通过键访问字典中的值
        # 【转换效果】转换前：config["model_params"]["sh_degree"] 
        #           转换后：model_params.sh_degree（可以用点操作访问属性）
        # 【语法3：变量赋值】variable = value - 将函数返回值赋给变量
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])

        # 【语法4：元组解包赋值】将右侧元组的值依次赋给左侧的多个变量
        # 【语法说明】a, b, c = (x, y, z) - 元组解包，将x赋给a，y赋给b，z赋给c
        # 【语法5：属性访问】self.xxx - 访问实例属性（self代表当前对象实例）
        # 【语法说明】self.attribute - 在类方法中访问实例变量
        # 【语法6：多行表达式】使用括号包裹可以换行写，提高可读性
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,      # 第一个值赋给 self.model_params
            opt_params,        # 第二个值赋给 self.opt_params
            pipeline_params,   # 第三个值赋给 self.pipeline_params
        )

        # ============ 知识点9：条件配置 ============
        # 【语法1：嵌套字典访问】self.config["Dataset"]["type"] - 访问嵌套字典的值
        # 【语法说明】dict["key1"]["key2"] - 逐层访问嵌套字典
        # 【语法2：比较运算符】== - 比较两个值是否相等，返回True或False（布尔值）
        # 【语法说明】value1 == value2 - 相等性比较，结果为布尔类型
        # 【语法3：布尔值赋值】variable = expression - 将表达式结果（True/False）赋给变量
        self.live_mode = self.config["Dataset"]["type"] == "realsense"  # 判断是否为实时模式
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"  # 判断是否单目
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]  # 获取是否使用球谐函数
        self.use_gui = self.config["Results"]["use_gui"]  # 获取是否使用GUI

        # 【语法4：if条件语句】if condition: - 如果条件为True则执行代码块
        # 【语法说明】if expression: - expression为True时执行缩进的代码块
        # 【语法5：布尔字面量】True - Python的布尔值（真值）
        if self.live_mode:              # 如果self.live_mode为True
            self.use_gui = True         # 则强制启用GUI（覆盖之前的配置）

        # 【语法6：直接赋值】从配置中读取值并保存到实例变量
        self.eval_rendering = self.config["Results"]["eval_rendering"]      # 评估渲染标志
        self.color_refinement = self.config["Results"]["color_refinement"]  # 颜色细化标志
        self.global_BA = self.config["Results"]["global_BA"]                # 全局BA标志

        # ============ 知识点10：模型初始化 ============
        # 【语法1：三元条件表达式】value1 if condition else value2 - 条件为True返回value1，否则返回value2
        # 【语法说明】这是if-else的简写形式：condition ? value1 : value2（其他语言）
        #           等价于：value1 if condition else value2（Python语法）
        # 【语法2：属性赋值】object.attribute = value - 给对象的属性赋值
        # 【语法说明】由于model_params是munch对象，可以用点操作访问和赋值
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0
        # 含义：如果使用球谐函数，sh_degree=3；否则sh_degree=0

        # 【语法3：类实例化】ClassName(arg1, arg2=value) - 创建类的实例
        # 【语法说明】第一个参数是位置参数，第二个是关键字参数
        # 【语法4：位置参数】model_params.sh_degree - 作为第一个参数传递
        # 【语法5：关键字参数】config=self.config - 明确指定参数名为config
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        # 创建高斯模型对象，传入球谐阶数和配置字典

        # 【语法6：方法调用】object.method(arg) - 在对象上调用方法
        # 【语法说明】self.gaussians是GaussianModel的实例，init_lr是它的方法
        self.gaussians.init_lr(self.config["opt_params"]["init_lr"])
        # 调用init_lr方法初始化学习率，传入配置中的初始学习率值

        # ============ 知识点11：数据集加载 ============
        # 【语法1：函数调用】function_name(arg1, arg2, arg3=value) - 调用函数
        # 【语法说明】load_dataset是从utils.dataset模块导入的函数
        # 【语法2：多行函数调用】函数参数可以换行写，提高可读性
        # 【语法3：位置参数】model_params, model_params.source_path - 前两个参数
        # 【语法4：关键字参数】config=config - 第三个参数使用关键字形式
        # 【语法5：属性访问】model_params.source_path - 通过点操作访问munch对象的属性
        self.dataset = load_dataset(
            model_params,              # 第一个参数：模型参数字典
            model_params.source_path,  # 第二个参数：数据源路径（属性访问）
            config=config              # 第三个参数：配置字典（关键字参数）
        )

        # ============ 知识点12：训练设置 ============
        # 【语法：方法调用】object.method(参数) - 调用对象的方法
        # 【语法说明】self.gaussians是GaussianModel的实例
        #           training_setup是方法名，opt_params是参数（munch对象）
        #           这个方法用于配置高斯模型的训练参数（优化器、学习率等）
        self.gaussians.training_setup(opt_params)  # 配置高斯模型的训练参数

        # ============ 知识点13：背景颜色设置 ============
        # 【语法1：列表字面量】[0, 0, 0] - 创建一个包含3个元素的列表
        # 【语法说明】[element1, element2, element3] - 列表用方括号，元素用逗号分隔
        bg_color = [0, 0, 0]  # RGB颜色值（红、绿、蓝都是0，表示黑色）

        # 【语法2：类方法调用】torch.tensor() - 调用torch模块的tensor函数（类方法）
        # 【语法说明】torch.tensor(data, dtype=类型, device="设备") - 创建张量
        # 【语法3：关键字参数】dtype=torch.float32 - 指定数据类型为32位浮点数
        # 【语法4：字符串字面量】"cuda" - 字符串用引号包裹，表示使用GPU
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # 将Python列表转换为PyTorch张量，指定数据类型和设备（GPU）

        # ============ 知识点14：多进程通信队列 ============
        # 【语法1：模块别名访问】mp.Queue() - 通过别名mp访问Queue类
        # 【语法说明】mp是torch.multiprocessing的别名（在导入时定义：as mp）
        # 【语法2：类实例化】Queue() - 创建队列对象（用于进程间通信）
        frontend_queue = mp.Queue()  # 前端到后端的队列（传递跟踪结果）
        backend_queue = mp.Queue()   # 后端到前端的队列（传递优化结果）

        # 【语法3：三元条件表达式】value1 if condition else value2
        # 【语法说明】如果self.use_gui为True，创建真实队列；否则创建假队列（不启用GUI时）
        # 【语法4：条件实例化】根据条件决定创建哪个类的实例
        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()  # 主进程到可视化队列
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()  # 可视化到主进程队列

        # ============ 知识点15：配置更新 ============
        # 【语法：嵌套字典赋值】dict["key1"]["key2"] = value - 修改嵌套字典的值
        # 【语法说明】通过多层方括号访问嵌套字典，然后赋值修改
        #           这是修改字典中嵌套值的标准语法
        self.config["Results"]["save_dir"] = save_dir         # 更新保存目录配置
        self.config["Training"]["monocular"] = self.monocular # 更新单目标志配置

        # ============ 知识点16：前后端初始化 ============
        # 【语法：类实例化】ClassName(arg1, arg2, arg3) - 调用类的构造函数
        # 【语法说明】FrontEnd和BackEnd是从utils.slam_frontend/backend导入的类
        #           传入参数创建类的实例，存储在self.frontend/backend中
        self.frontend = FrontEnd(self.config, mast3r_model, self.save_dir)  # 创建前端对象
        self.backend = BackEnd(self.config, self.save_dir)                   # 创建后端对象

        # ============ 知识点17：前后端连接 ============
        # 【语法：对象属性赋值】object.attribute = value - 给对象的属性赋值
        # 【语法说明】self.frontend是FrontEnd类的实例，通过点操作设置其属性
        #           这些属性在类中可能没有在__init__中初始化，但可以在运行时动态添加
        self.frontend.dataset = self.dataset                    # 将数据集对象赋给前端
        self.frontend.background = self.background              # 将背景张量赋给前端
        self.frontend.pipeline_params = self.pipeline_params    # 将管道参数赋给前端

        # 设置前端和后端的通信队列（双向通信）
        self.frontend.frontend_queue = frontend_queue  # 前端到后端的队列
        self.frontend.backend_queue = backend_queue    # 后端到前端的队列
        self.frontend.q_main2vis = q_main2vis          # 主进程到可视化队列
        self.frontend.q_vis2main = q_vis2main          # 可视化到主进程队列

        # 【语法：无参数方法调用】object.method() - 调用对象的方法
        # 【语法说明】set_hyperparams()是FrontEnd类的方法，不需要参数
        self.frontend.set_hyperparams()  # 调用方法设置前端的超参数

        # ============ 知识点18：后端配置 ============
        # 【语法：对象属性赋值】给后端对象的各个属性赋值
        self.backend.gaussians = self.gaussians                    # 高斯模型对象
        self.backend.background = self.background                  # 背景颜色张量
        self.backend.cameras_extent = 6.0                          # 浮点数字面量
        self.backend.pipeline_params = self.pipeline_params        # 管道参数字典
        self.backend.opt_params = self.opt_params                  # 优化参数字典

        # 设置后端通信队列和配置参数
        self.backend.frontend_queue = frontend_queue  # 前端到后端队列
        self.backend.backend_queue = backend_queue    # 后端到前端队列
        self.backend.live_mode = self.live_mode       # 实时模式标志（布尔值）
        # 【语法：len()函数】len(iterable) - 返回可迭代对象的长度（元素个数）
        # 【语法说明】self.dataset是可迭代对象（如列表），len()返回其元素数量
        self.backend.num_frames = len(self.dataset)   # 计算数据集帧数

        # 【语法：方法调用】调用后端的方法设置超参数
        self.backend.set_hyperparams()

        # ============ 知识点19：GUI参数界面 ============
        # 【语法1：模块属性访问】gui_utils.ParamsGUI - 通过点操作访问模块中的类
        # 【语法说明】gui_utils是从gui模块导入的子模块，ParamsGUI是其中的类
        # 【语法2：关键字参数】key=value - 所有参数都使用关键字形式传递
        # 【语法说明】使用关键字参数可以让代码更清晰，不依赖参数顺序
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,      # 关键字参数：渲染管道参数
            background=self.background,     # 关键字参数：背景颜色张量
            gaussians=self.gaussians,       # 关键字参数：高斯模型对象
            q_main2vis=q_main2vis,          # 关键字参数：主进程到可视化队列
            q_vis2main=q_vis2main,          # 关键字参数：可视化到主进程队列
        )

        # ============ 知识点20：多进程启动 ============
        # 【语法1：Process类实例化】mp.Process(target=函数名) - 创建进程对象
        # 【语法说明】target参数指定进程要执行的函数（这里是方法）
        # 【语法2：方法引用】self.backend.run - 引用对象的方法（不加括号）
        # 【语法说明】不加括号是方法引用，加括号是方法调用
        backend_process = mp.Process(target=self.backend.run)  # 创建后端进程对象

        # 【语法3：if条件语句】if condition: - 条件为True时执行代码块
        if self.use_gui:
            # 【语法4：元组作为参数】args=(arg1,) - 单元素元组需要加逗号
            # 【语法说明】args参数必须是元组，单元素元组写成(arg,)而不是(arg)
            # 【语法5：函数引用】slam_gui.run - 引用模块中的函数
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            # 【语法6：方法调用】object.start() - 启动进程
            gui_process.start()  # 启动GUI进程（开始执行target指定的函数）
            # 【语法7：模块函数调用】time.sleep(秒数) - 让当前线程休眠指定秒数
            time.sleep(5)  # 等待5秒，让GUI进程有时间启动完成

        # 启动后端进程（在另一个进程中执行self.backend.run方法）
        backend_process.start()

        # ============ 知识点21：前端运行（主线程） ============
        # 【语法：方法调用】object.method() - 在主线程中调用方法
        # 【语法说明】run()方法会阻塞当前线程，直到前端处理完成
        self.frontend.run()  # 前端在主线程中运行（阻塞式）

        # 【语法：队列方法调用】queue.put(item) - 向队列中添加元素
        # 【语法说明】backend_queue是mp.Queue对象，put方法向队列发送数据
        # 【语法：列表字面量】["pause"] - 创建包含一个字符串元素的列表
        backend_queue.put(["pause"])  # 向后端队列发送暂停信号（列表格式）

        # ============ 知识点22：性能统计 ============
        # 【语法：方法调用】object.record() - 记录CUDA事件的时间点
        end.record()  # 记录结束时间点

        # 【语法：模块函数调用】torch.cuda.synchronize() - 同步GPU操作
        # 【语法说明】等待所有GPU操作完成，确保时间测量的准确性
        torch.cuda.synchronize()  # 等待GPU操作完成

        # 【语法1：属性访问】self.frontend.cameras - 访问对象的属性
        # 【语法2：len()函数】len(object) - 获取列表/对象长度
        N_frames = len(self.frontend.cameras)  # 获取相机数量（帧数）

        # 【语法1：方法调用】object.method() - 计算两个事件之间的时间差（毫秒）
        # 【语法2：算术运算】* 0.001 - 乘法运算，将毫秒转换为秒
        # 【语法3：除法运算】/ - 除法运算符
        # 【语法4：括号优先级】() - 括号内的运算先执行
        FPS = N_frames / (start.elapsed_time(end) * 0.001)  # 计算FPS（帧/秒）

        # 【语法：函数调用】Log(参数1, 参数2, tag=参数3) - 自定义日志函数
        # 【语法说明】Log是从utils.logging_utils导入的函数，用于输出日志
        # 【语法：关键字参数】tag="Eval" - 使用关键字参数指定标签
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")  # 输出总时间
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")  # 输出FPS

        # ============ 知识点23：评估和渲染 ============
        # 【语法：if条件语句】if condition: - 条件为True时执行代码块
        if self.eval_rendering:  # 如果启用评估渲染
            # 【语法：属性访问】self.frontend.xxx - 访问前端对象的属性
            self.gaussians = self.frontend.gaussians    # 获取前端的高斯模型
            kf_indices = self.frontend.kf_indices       # 获取关键帧索引列表

            # ============ 知识点24：绝对轨迹误差评估 ============
            # 【语法：函数调用】function_name(位置参数, 关键字参数)
            # 【语法说明】eval_ate是从utils.eval_utils导入的函数
            # 【语法：混合参数】前几个是位置参数，后面是关键字参数
            ATE = eval_ate(
                self.frontend.cameras,      # 位置参数1：相机列表
                self.frontend.kf_indices,   # 位置参数2：关键帧索引
                self.save_dir,              # 位置参数3：保存目录路径
                0,                          # 位置参数4：起始帧索引（整数）
                final=True,                 # 关键字参数：是否最终评估（布尔值）
                monocular=self.monocular,   # 关键字参数：是否单目（布尔值）
            )

            # ============ 知识点25：渲染评估 ============
            # 【语法：多参数函数调用】函数有很多参数，使用多行格式提高可读性
            # 【语法说明】eval_rendering是从utils.eval_utils导入的函数
            rendering_result = eval_rendering(
                self.frontend.cameras,                    # 位置参数1：相机列表
                self.gaussians,                           # 位置参数2：高斯模型对象
                self.dataset,                             # 位置参数3：数据集对象
                self.save_dir,                            # 位置参数4：保存目录
                self.pipeline_params,                     # 位置参数5：管道参数
                self.background,                          # 位置参数6：背景张量
                datatype=self.config["Dataset"]["type"],  # 关键字参数：数据集类型字符串
                kf_indices=kf_indices,                    # 关键字参数：关键帧索引
                iteration="before_opt",                   # 关键字参数：迭代标识字符串
            )

# ============ 知识点26：主函数 ============
# 【语法：函数定义】def function_name(): - 定义一个函数
# 【语法说明】def是关键字，main是函数名，()内是参数列表（这里无参数）
#           函数名通常使用小写字母和下划线（蛇形命名法）
def main():
    """
    主函数：程序入口点

    学习目标：
    - 理解命令行参数解析
    - 理解配置文件加载
    - 理解模型初始化
    - 理解系统启动流程
    """

    # ============ 知识点27：命令行参数解析 ============
    # 【语法1：类实例化】ArgumentParser(description="字符串") - 创建参数解析器对象
    # 【语法说明】ArgumentParser是从argparse模块导入的类
    # 【语法2：关键字参数】description= - 程序的描述信息
    parser = ArgumentParser(description="S3PO Baseline SLAM System")

    # 【语法1：方法调用】object.add_argument() - 添加命令行参数定义
    # 【语法2：字符串字面量】"--config" - 参数名（双短横线表示长参数名）
    # 【语法3：关键字参数】required=True - 参数是否为必需（True表示必需）
    # 【语法4：关键字参数】help="字符串" - 参数的帮助信息
    parser.add_argument("--config", required=True, help="Path to config file")
    # 【语法5：关键字参数】default=None - 参数的默认值（None表示没有默认值）
    parser.add_argument("--save_dir", default=None, help="Save directory")
    # 【语法6：多行表达式】如果字符串太长，可以换行写
    parser.add_argument("--mast3r_model", default="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                       help="MASt3R model name")

    # 【语法：方法调用】object.parse_args() - 解析命令行参数
    # 【语法说明】解析命令行传入的参数，返回包含参数值的对象
    args = parser.parse_args()  # args是一个对象，可以通过args.参数名访问

    # ============ 知识点28：配置加载 ============
    # 【语法1：属性访问】args.config - 访问args对象的config属性
    # 【语法说明】args.config是命令行传入的配置文件路径
    # 【语法2：函数调用】load_config(路径) - 调用函数加载配置文件
    # 【语法说明】load_config是从utils.config_utils导入的函数
    config = load_config(args.config)  # 加载并返回配置字典

    # ============ 知识点29：保存目录设置 ============
    # 【语法1：is比较运算符】is None - 检查变量是否为None
    # 【语法说明】is用于身份比较（是否为同一个对象），None用is判断
    if args.save_dir is None:  # 如果保存目录参数未指定
        # 【语法1：类方法调用】datetime.now() - 调用datetime类的now方法
        # 【语法说明】datetime.now()返回当前日期时间的datetime对象
        # 【语法2：方法调用】object.strftime(格式字符串) - 格式化日期时间为字符串
        # 【语法说明】"%Y-%m-%d-%H-%M-%S"是格式字符串，%Y=年，%m=月，%d=日等
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # 【语法1：f-string格式化】f"字符串{变量}" - 在字符串中插入变量值
        # 【语法说明】f-string是Python 3.6+的字符串格式化方式
        # 【语法2：嵌套字典访问】config['Dataset']['type'] - 访问嵌套字典
        # 【语法3：字符串连接】_ - 下划线用于连接字符串部分
        args.save_dir = f"results/{config['Dataset']['type']}_{config['Dataset']['sequence']}_{current_time}"

    # ============ 知识点30：模型加载 ============
    # 【语法1：类方法调用】ClassName.from_pretrained(参数) - 调用类方法
    # 【语法说明】from_pretrained是类方法，用于从预训练模型加载
    # 【语法2：方法链式调用】.from_pretrained().to() - 连续调用方法
    # 【语法说明】from_pretrained返回模型对象，继续调用.to()方法移动到GPU
    # 【语法3：方法调用】.to("cuda") - 将模型移动到GPU设备
    mast3r_model = AsymmetricMASt3R.from_pretrained(args.mast3r_model).to("cuda")

    # ============ 知识点31：创建目录 ============
    # 【语法：函数调用】mkdir_p(路径) - 创建目录（如果不存在）
    # 【语法说明】mkdir_p是从gaussian_splatting.utils.system_utils导入的函数
    mkdir_p(args.save_dir)  # 确保保存目录存在，不存在则创建

    # ============ 知识点32：启动SLAM系统 ============
    # 【语法：类实例化】ClassName(arg1, arg2, arg3) - 创建类的实例
    # 【语法说明】SLAM是前面定义的类，传入配置、模型、保存目录初始化
    #           这会调用SLAM类的__init__方法，启动整个SLAM系统
    slam = SLAM(config, mast3r_model, args.save_dir)

# ============ 知识点33：程序入口判断 ============
# 【语法1：特殊变量】__name__ - Python的特殊变量
# 【语法说明】当直接运行文件时，__name__的值为"__main__"
#           当文件被导入时，__name__的值为模块名
# 【语法2：字符串比较】== "__main__" - 比较字符串是否相等
# 【语法3：if条件语句】if condition: - 条件为True时执行代码块
# 【语法说明】这个判断确保只有直接运行此文件时才执行main函数
#           如果文件被其他文件导入，不会执行main函数
if __name__ == "__main__":
    main()  # 调用main函数，程序从这里开始执行

# ============ 练习题 ============
"""
练习1：理解系统架构
请回答：
1. SLAM系统中前后端分离有什么好处？
2. 为什么需要使用多进程而不是多线程？
3. GUI进程为什么需要等待5秒启动？

练习2：理解通信机制
请回答：
1. frontend_queue 和 backend_queue 各用于什么通信？
2. q_main2vis 和 q_vis2main 的作用是什么？

练习3：修改代码
请修改代码，添加一个命令行参数来控制是否启用性能计时

练习4：扩展功能
请设计如何添加一个新的评估指标到SLAM系统中
"""

if __name__ == "__main__":
    print("这是一个SLAM主程序的学习版本")
    print("主要知识点：多进程架构、前后端分离、队列通信、系统集成")
    print("运行命令：python 05_slam_main_学习版.py --config your_config.yaml")

