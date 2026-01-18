# -*- coding: utf-8 -*-
"""
文件：06_self_概念详解.py
功能：专门解释Python中self的概念
难度：⭐⭐（中等，重要概念）
学习目标：
1. 理解self是什么
2. 理解self为什么必须存在
3. 理解实例变量和方法
4. 理解对象和类的关系

这个文件的作用是：通过简单示例让你彻底理解self的概念
"""

# ============ 知识点1：什么是self？ ============
"""
self 是Python面向对象编程中的一个特殊参数，它代表"当前对象"

举个现实世界的例子：
假设我们有一班学生，每个学生都有自己的名字和成绩。
如果没有self，那么所有学生都会共享同一个名字和成绩。
有了self，每个学生对象都能保存自己的数据。
"""

# ============ 知识点2：没有self会怎么样？ ============
class StudentWithoutSelf:
    """没有使用self的错误示例"""
    def __init__(name, score):  # 错误：没有self参数
        name = name  # 这会创建一个局部变量，而不是对象属性
        score = score

    def get_info(name):  # 错误：没有self参数
        return "学生：" + name

# 创建对象测试
# student1 = StudentWithoutSelf("小明", 95)  # 这会报错，因为__init__需要self参数

# ============ 知识点3：正确的self使用 ============
class Student:
    """正确使用self的类"""

    def __init__(self, name, score):
        """
        构造函数：创建学生对象时自动调用

        参数说明：
        - self: 当前创建的学生对象（自动传入）
        - name: 学生姓名（需要手动传入）
        - score: 学生成绩（需要手动传入）
        """
        # self.name：给当前学生对象添加一个name属性
        self.name = name
        # self.score：给当前学生对象添加一个score属性
        self.score = score

        print("创建了学生对象：" + self.name)  # 可以在__init__中使用self

    def get_info(self):
        """
        获取学生信息的方法

        参数说明：
        - self: 当前学生对象（自动传入）
        返回值：包含学生信息的字符串
        """
        # 在方法中使用self来访问对象的属性
        return "学生：" + self.name + "，成绩：" + str(self.score)

    def update_score(self, new_score):
        """更新学生成绩"""
        self.score = new_score
        print(self.name + "的成绩更新为：" + str(self.score))

# ============ 知识点4：创建对象和使用self ============
def demonstrate_self():
    """演示self的概念和用法"""

    print("=== 创建学生对象 ===")

    # 创建第一个学生对象
    student1 = Student("小明", 95)
    print("student1对象的name属性：" + student1.name)
    print("student1对象的score属性：" + str(student1.score))
    print("调用student1的方法：" + student1.get_info())
    print()

    # 创建第二个学生对象
    student2 = Student("小红", 88)
    print("student2对象的name属性：" + student2.name)
    print("student2对象的score属性：" + str(student2.score))
    print("调用student2的方法：" + student2.get_info())
    print()

    # 修改对象属性
    print("=== 修改对象属性 ===")
    student1.update_score(98)  # student1的成绩变为98
    student2.update_score(92)  # student2的成绩变为92

    print("修改后student1：" + student1.get_info())
    print("修改后student2：" + student2.get_info())
    print()

    # 证明每个对象都有独立的属性
    print("=== 证明对象独立性 ===")
    print("student1.name = '" + student1.name + "'，student2.name = '" + student2.name + "'")
    print("两个对象有不同的name属性，互不影响！")

# ============ 知识点5：self在dataset中的应用 ============
class DatasetParserDemo:
    """模拟dataset中的self使用"""

    def __init__(self, dataset_name, data_path):
        """初始化数据集解析器"""
        # self保存了这个解析器对象的配置
        self.dataset_name = dataset_name  # 每个解析器有自己的数据集名称
        self.data_path = data_path       # 每个解析器有自己的数据路径
        self.is_loaded = False          # 每个解析器有自己的加载状态

        print("创建了" + self.dataset_name + "数据集解析器")

    def load_data(self):
        """加载数据"""
        # 在方法中使用self访问对象的属性
        print("正在加载" + self.dataset_name + "数据集...")
        print("数据路径：" + self.data_path)
        self.is_loaded = True  # 修改对象的状态
        return self.dataset_name + "数据加载完成"

    def get_status(self):
        """获取加载状态"""
        status = "已加载" if self.is_loaded else "未加载"
        return self.dataset_name + "状态：" + status

def demonstrate_dataset_self():
    """演示dataset中的self概念"""

    print("=== 数据集解析器示例 ===")

    # 创建两个不同的数据集解析器
    parser1 = DatasetParserDemo("KITTI", "/data/kitti")
    parser2 = DatasetParserDemo("dl3dv", "/data/dl3dv")

    print()
    print("=== 调用方法 ===")
    result1 = parser1.load_data()  # parser1.load_data() 相当于 DatasetParserDemo.load_data(parser1)
    print(result1)
    print()

    result2 = parser2.load_data()  # parser2.load_data() 相当于 DatasetParserDemo.load_data(parser2)
    print(result2)
    print()

    print("=== 检查状态 ===")
    print(parser1.get_status())  # parser1的状态
    print(parser2.get_status())  # parser2的状态
    print()

    print("=== 理解self的本质 ===")
    print("当我们调用 parser1.load_data() 时，Python自动将parser1作为self参数传入")
    print("相当于：DatasetParserDemo.load_data(parser1)")
    print("这就是为什么方法定义时需要self参数，而调用时不需要传入self")

# ============ 练习题 ============
"""
练习1：理解self
请回答：
1. 为什么Python方法必须有self参数？
2. 如果没有self会发生什么？
3. 每个对象如何保存自己的数据？

练习2：修改代码
请为Student类添加一个新的方法：add_score(delta)，用来增加分数

练习3：创建自己的类
请创建一个Book（图书）类，包含title（标题）和author（作者）属性，
并实现get_info()方法
"""

# ============ 主程序 ============
if __name__ == "__main__":
    print("=== 理解self概念 ===")
    print()

    demonstrate_self()
    print("\n" + "="*50 + "\n")

    demonstrate_dataset_self()
    print("\n" + "="*50 + "\n")

    print("=== 练习建议 ===")
    print("1. 仔细观察每个方法调用时，self是如何指向不同对象的")
    print("2. 理解对象属性是独立的，每个对象都有自己的副本")
    print("3. 记住：self就是当前对象的引用")

    # 练习：创建一个Book类
    print("\n=== 练习：创建Book类 ===")

    class Book:
        def __init__(self, title, author):
            self.title = title
            self.author = author

        def get_info(self):
            return "《" + self.title + "》 - " + self.author

    # 测试Book类
    book1 = Book("Python编程", "张三")
    book2 = Book("机器学习", "李四")

    print(book1.get_info())
    print(book2.get_info())
