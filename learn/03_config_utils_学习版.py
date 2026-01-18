"""
文件：03_config_utils_学习版.py
功能：学习配置文件加载工具模块
难度：⭐⭐⭐（中等）
学习目标：
1. 理解文件读取（with open）
2. 理解YAML文件格式
3. 理解字典操作（get、items、keys）
4. 理解递归函数
5. 理解类型检查（isinstance）
6. 理解条件判断和逻辑分支
7. 理解函数嵌套调用

这个文件的作用是：加载YAML配置文件，支持配置文件继承和递归合并
"""

# ============ 知识点1：导入YAML库 ============
# yaml 是一个第三方库，用于解析YAML格式的配置文件
# YAML是一种人类可读的数据序列化格式，常用于配置文件
import yaml


def load_config(path, default_path=None):
    """
    加载配置文件
    
    知识点：函数文档字符串
    - Args: 参数说明
    - Returns: 返回值说明
    
    这个函数支持：
    1. 加载指定路径的配置文件
    2. 支持配置文件继承（inherit_from）
    3. 支持默认配置（default_path）
    4. 合并配置（继承的配置 + 特殊配置）
    
    参数：
        path: 配置文件的路径（字符串）
              例如："./configs/my_config.yaml"
        
        default_path: 默认配置文件的路径（可选）
                     如果提供了，并且当前配置文件没有inherit_from，
                     则使用这个默认配置文件
    
    返回值：
        cfg: 配置字典（dict），包含所有配置项
    
    配置文件示例（config.yaml）：
        # 方式1：继承其他配置
        inherit_from: "./configs/base_config.yaml"
        model_params:
            learning_rate: 0.01
        
        # 方式2：独立配置
        model_params:
            learning_rate: 0.01
            batch_size: 32
    """
    # ============ 知识点2：文件读取（with语句） ============
    # with open(...) as f: 是Python推荐的文件读取方式
    # 优点：自动关闭文件，即使发生异常也会关闭
    # "r" 表示只读模式（read mode）
    # as f 表示把文件对象赋值给变量 f
    
    # 打开并读取指定路径的配置文件
    with open(path, "r") as f:
        # yaml.full_load(f) 读取YAML文件内容，转换为Python字典
        # 例如：YAML文件中的 key: value 会变成字典的 {"key": "value"}
        cfg_special = yaml.full_load(f)

    # ============ 知识点3：字典的get方法 ============
    # cfg_special.get("inherit_from") 获取字典中的"inherit_from"键的值
    # get() 方法的优点：如果键不存在，返回 None（不会报错）
    # 对比：cfg_special["inherit_from"] 如果键不存在会报 KeyError
    inherit_from = cfg_special.get("inherit_from")

    # ============ 知识点4：条件判断和递归调用 ============
    # 如果配置文件中有 inherit_from 字段，说明要继承其他配置文件
    if inherit_from is not None:
        # 递归调用自己，先加载被继承的配置文件
        # 这会产生一个配置的"继承链"
        # 例如：A继承B，B继承C，那么先加载C，再加载B，最后加载A
        cfg = load_config(inherit_from, default_path)
    
    # ============ 知识点5：elif和else分支 ============
    # elif: else if 的缩写，表示"否则如果"
    # else: 表示"否则"
    elif default_path is not None:
        # 如果没有继承，但有默认配置文件，加载默认配置
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        # 既没有继承，也没有默认配置，创建一个空字典
        # dict() 等同于 {}
        cfg = dict()

    # ============ 知识点6：调用其他函数 ============
    # 调用下面的 update_recursive 函数，合并配置
    # cfg_special 中的配置会覆盖 cfg 中相同键的值
    update_recursive(cfg, cfg_special)

    # 返回合并后的配置字典
    return cfg


def update_recursive(dict1, dict2):
    """
    递归更新字典
    
    知识点：递归函数
    - 函数调用自己
    - 必须有终止条件（base case）
    - 用于处理嵌套结构（如嵌套字典）
    
    功能：
    - dict2 的配置会覆盖 dict1 中相同键的值
    - 如果是嵌套字典，会递归更新
    - 最终结果保存在 dict1 中（dict1会被修改）
    
    参数：
        dict1: 第一个字典（会被修改）
        dict2: 第二个字典（提供更新值）
    
    示例：
        dict1 = {
            "a": 1,
            "b": {"x": 10, "y": 20}
        }
        dict2 = {
            "a": 2,  # 覆盖 dict1["a"]
            "b": {"x": 15}  # 只覆盖 dict1["b"]["x"]，保留 dict1["b"]["y"]
        }
        update_recursive(dict1, dict2)
        # 结果：dict1 = {"a": 2, "b": {"x": 15, "y": 20}}
    """
    # ============ 知识点7：字典遍历 ============
    # dict2.items() 返回字典的 (键, 值) 对
    # for k, v in ... 循环遍历每一对键值
    # k 是键（key），v 是值（value）
    for k, v in dict2.items():
        # ============ 知识点8：字典键的检查 ============
        # k not in dict1 检查键 k 是否不在 dict1 中
        if k not in dict1:
            # 如果键不存在，在 dict1 中创建这个键，值为空字典
            # 这样后续才能安全地递归更新
            dict1[k] = dict()
        
        # ============ 知识点9：类型检查 ============
        # isinstance(v, dict) 检查 v 是否是字典类型
        # isinstance() 用于类型检查，返回 True 或 False
        if isinstance(v, dict):
            # ============ 知识点10：递归调用 ============
            # 如果值是字典，递归调用自己，更新嵌套的字典
            # 这是递归的"继续"部分：处理更深一层的嵌套
            # dict1[k] 是 dict1 中键 k 对应的值（应该也是字典）
            # v 是 dict2 中键 k 对应的值（也是字典）
            # 递归更新这两个字典
            update_recursive(dict1[k], v)
        else:
            # ============ 知识点11：终止条件 ============
            # 如果值不是字典（是普通值，如数字、字符串等），直接覆盖
            # 这是递归的"终止"部分：当值不是字典时，不再递归
            dict1[k] = v
    
    # 注意：这个函数没有return语句
    # 因为直接修改了dict1，不需要返回值（Python中可以修改可变对象）


# ============ 测试示例 ============
if __name__ == "__main__":
    # 测试递归更新功能
    
    print("=== 测试递归字典更新功能 ===")
    
    # 创建测试字典
    base_config = {
        "model": {
            "name": "ResNet",
            "layers": 50,
            "optimizer": {
                "type": "Adam",
                "lr": 0.001
            }
        },
        "dataset": {
            "batch_size": 32
        }
    }
    
    special_config = {
        "model": {
            "layers": 101,  # 覆盖 layers: 50 → 101
            "optimizer": {
                "lr": 0.01  # 覆盖 lr: 0.001 → 0.01，保留 type: "Adam"
            }
        },
        "new_key": "new_value"  # 添加新键
    }
    
    print("原始配置：")
    print(base_config)
    
    # 执行更新
    update_recursive(base_config, special_config)
    
    print("\n更新后的配置：")
    print(base_config)
    print("\n可以看到：")
    print("- layers 被更新为 101")
    print("- optimizer.lr 被更新为 0.01")
    print("- optimizer.type 保持不变（仍然是 Adam）")
    print("- model.name 保持不变（仍然是 ResNet）")
    print("- new_key 被添加")
