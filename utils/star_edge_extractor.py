"""
STAR-Edge轮廓提取模块
用于从MASt3R生成的点云中提取边缘点
"""

import sys
import os

# 设置 LD_LIBRARY_PATH 以确保能找到 fftw 库
# 注意：Python 启动后设置环境变量可能无效，需要使用 ctypes
fftw_lib_path = '/usr/lib/x86_64-linux-gnu'
if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = fftw_lib_path
elif fftw_lib_path not in os.environ['LD_LIBRARY_PATH']:
    os.environ['LD_LIBRARY_PATH'] = fftw_lib_path + ':' + os.environ['LD_LIBRARY_PATH']

# 使用 ctypes 设置库路径（在导入 .so 文件之前）
try:
    import ctypes
    # 尝试加载 fftw 库，确保路径正确
    try:
        ctypes.CDLL(os.path.join(fftw_lib_path, 'libfftw3.so.3.6.9'), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        # 如果 3.6.9 不存在，尝试 3.5.8
        try:
            ctypes.CDLL(os.path.join(fftw_lib_path, 'libfftw3.so.3.5.8'), mode=ctypes.RTLD_GLOBAL)
        except:
            pass
except:
    pass  # 如果 ctypes 设置失败，继续尝试

import numpy as np
import torch
import time

# 添加STAR-Edge路径
# 尝试多个可能的路径
possible_paths = [
    os.path.join(os.path.dirname(__file__), '..', 'STAR-Edge'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'STAR-Edge'),
    'STAR-Edge',  # 当前目录
    '/workspace/STAR-Edge',  # Docker环境可能路径
]

star_edge_path = None
for path in possible_paths:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        star_edge_path = abs_path
        break

STAR_EDGE_AVAILABLE = False
if star_edge_path:
    # 将STAR-Edge根目录添加到路径（net.py在那里）
    if star_edge_path not in sys.path:
        sys.path.insert(0, star_edge_path)
    
    # 将pre_process目录添加到路径（.so文件在那里）
    pre_process_path = os.path.join(star_edge_path, 'pre_process')
    if os.path.exists(pre_process_path):
        sys.path.insert(0, pre_process_path)
    
    # 确保LocalSH目录不在路径中，避免导入不存在的__init__.py
    local_sh_dir = os.path.join(star_edge_path, 'LocalSH')
    if local_sh_dir in sys.path:
        sys.path.remove(local_sh_dir)
    
    # 清除可能的模块缓存
    for key in list(sys.modules.keys()):
        if key.startswith('LocalSH'):
            del sys.modules[key]
    
    try:
        # 在加载.so文件前，先预加载fftw库（解决依赖问题）
        import ctypes
        fftw_lib_path = '/usr/lib/x86_64-linux-gnu'
        try:
            # 尝试加载符号链接
            ctypes.CDLL(os.path.join(fftw_lib_path, 'libfftw3.so.3.6.9'), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # 如果符号链接不存在，尝试直接加载实际文件
            try:
                ctypes.CDLL(os.path.join(fftw_lib_path, 'libfftw3.so.3.5.8'), mode=ctypes.RTLD_GLOBAL)
            except:
                pass  # 如果都失败，继续尝试加载.so文件
        
        # 使用importlib直接加载.so文件
        import importlib.util
        import glob
        import sys
        
        # 查找.so文件
        so_files = glob.glob(os.path.join(pre_process_path, 'LocalSH*.so'))
        if not so_files:
            raise FileNotFoundError(f"未找到LocalSH.so文件在: {pre_process_path}")
        
        # 从文件名中提取Python版本，并选择匹配当前Python版本的.so文件
        python_version = sys.version_info
        python_version_str = f"{python_version.major}.{python_version.minor}"
        matching_so_file = None
        
        # 优先查找匹配当前Python版本的.so文件
        for so_file in so_files:
            if f"cpython-{python_version.major}{python_version.minor}" in so_file:
                matching_so_file = so_file
                break
        
        # 如果没找到完全匹配的，使用第一个（可能是兼容的）
        if matching_so_file is None:
            matching_so_file = so_files[0]
            # 从文件名提取编译时的Python版本
            import re
            match = re.search(r'cpython-(\d)(\d)', matching_so_file)
            if match:
                compiled_major = int(match.group(1))
                compiled_minor = int(match.group(2))
                if compiled_major != python_version.major or compiled_minor != python_version.minor:
                    raise ImportError(
                        f"Python版本不匹配: .so文件是为Python {compiled_major}.{compiled_minor}编译的，但当前Python版本是 {python_version.major}.{python_version.minor}。\n"
                        f"解决方案：\n"
                        f"1. 使用Python {compiled_major}.{compiled_minor}运行程序\n"
                        f"2. 重新编译LocalSH.so文件以匹配当前Python版本\n"
                        f"3. 暂时禁用轮廓提取功能（程序将继续运行，但不进行轮廓提取）"
                    )
        
        so_file = matching_so_file
        
        # 使用importlib加载.so文件
        spec = importlib.util.spec_from_file_location("LocalSH", so_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法从.so文件创建模块规范: {so_file}")
        
        LocalSH = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(LocalSH)
        
        # 访问子模块
        if not hasattr(LocalSH, 'LocalSHFeature'):
            raise AttributeError(f"LocalSH模块没有LocalSHFeature属性。可用属性: {[x for x in dir(LocalSH) if not x.startswith('_')]}")
        
        localsh_feature = LocalSH.LocalSHFeature
        normal_refine = LocalSH.NormalRefine
        
        if not hasattr(localsh_feature, 'ComLSHF_knn_upsample'):
            raise AttributeError("LocalSHFeature子模块没有ComLSHF_knn_upsample方法")
        
        # LocalSH导入成功，继续导入其他模块
        try:
            from net import DescClassifier
        except ImportError as e:
            print(f"[ERROR] 导入 net.DescClassifier 失败: {e}")
            print(f"[ERROR] net.py路径: {os.path.join(star_edge_path, 'net.py')}")
            raise
        
        # 所有模块导入成功
        STAR_EDGE_AVAILABLE = True
            
    except (ImportError, FileNotFoundError, AttributeError) as e:
        error_msg = str(e)
        # 检查是否是Python版本不匹配
        if "Python version mismatch" in error_msg or "Python版本不匹配" in error_msg:
            print(f"[WARNING] LocalSH模块无法加载: Python版本不匹配")
            # 尝试从错误信息中提取编译时的Python版本
            import re
            match = re.search(r'Python (\d)\.(\d)', error_msg)
            if match:
                compiled_version = f"{match.group(1)}.{match.group(2)}"
                print(f"[WARNING] - .so文件是为Python {compiled_version}编译的")
            else:
                print(f"[WARNING] - .so文件版本与当前Python不匹配")
            print(f"[WARNING] - 当前Python版本: {sys.version_info.major}.{sys.version_info.minor}")
            print(f"[WARNING] - 轮廓提取功能将被禁用，程序将继续运行")
            print(f"[WARNING] - 如需使用轮廓提取，请使用匹配的Python版本或重新编译.so文件")
        else:
            print(f"[ERROR] 导入 LocalSH 失败: {e}")
            print(f"[ERROR] STAR-Edge路径: {star_edge_path}")
            print(f"[ERROR] pre_process路径: {pre_process_path}")
            import traceback
            traceback.print_exc()
        
        # 不抛出异常，让程序继续运行（轮廓提取功能将被禁用）
        STAR_EDGE_AVAILABLE = False
        localsh_feature = None
        normal_refine = None
    except ImportError as e:
        print(f"[ERROR] STAR-Edge模块导入失败")
        print(f"[ERROR] 详细错误: {e}")
        print(f"[ERROR] Python路径: {sys.path[:3]}")
        import traceback
        traceback.print_exc()
        STAR_EDGE_AVAILABLE = False
else:
        print(f"[ERROR] STAR-Edge路径不存在")
        print(f"[ERROR] 尝试的路径: {possible_paths}")
        STAR_EDGE_AVAILABLE = False


class STAREdgeExtractor:
    """STAR-Edge轮廓提取器"""
    
    def __init__(self, model_path=None, bw=10, kk=26, sampleNum=40, mu=0.1, verbose=True, max_points=70000):
        """
        初始化STAR-Edge提取器
        
        Args:
            model_path: 模型权重路径，如果为None则使用默认路径
            bw: 带宽参数（默认10）
            kk: K近邻数量（默认26）
            sampleNum: 采样数量（默认40，即bw*4）
            mu: 边缘细化参数（默认0.1）
            verbose: 是否输出详细信息
            max_points: STAR-Edge处理时的最大点数（默认70000）
                        仅在STAR-Edge内部临时下采样，不影响用于渲染的点云密度
        """
        self.bw = bw
        self.kk = kk
        self.sampleNum = sampleNum
        self.mu = mu
        self.verbose = verbose
        self.max_points = max_points  # STAR-Edge内部临时下采样的目标点数
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if not STAR_EDGE_AVAILABLE:
            # 获取star_edge_path（可能在全局作用域中）
            try:
                edge_path = star_edge_path if 'star_edge_path' in globals() else '未知'
            except:
                edge_path = '未知'
            
            error_msg = (
                "STAR-Edge模块不可用，请检查：\n"
                f"1. STAR-Edge路径是否存在: {edge_path}\n"
                "2. LocalSH模块是否已编译（需要LocalSH.cpython-*.so文件）\n"
                "3. 所有依赖是否已安装（PyTorch, numpy等）\n"
                "4. 模型文件是否存在（STAR-Edge/model/best.ckpt）"
            )
            raise RuntimeError(error_msg)
        
        # 设置模型路径
        if model_path is None:
            # star_edge_path在模块级别定义，这里应该可以访问
            model_path = os.path.join(star_edge_path, "model", "best.ckpt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"STAR-Edge模型文件不存在: {model_path}")
        
        # 加载分类网络
        self.net = DescClassifier()
        self.net.load(model_path)
        self.net.to(self.device)
        self.net.eval()
    
    def extract_edges(self, points, refine=True, return_details=False):
        """
        提取点云中的边缘点
        
        Args:
            points: 点云 [N, 3] (numpy array, float32或float64)
            refine: 是否进行边缘点细化（默认True）
            return_details: 是否返回详细信息（默认False）
        
        Returns:
            edge_mask: 边缘点标记 [N] (boolean array)
            edge_points: 边缘点云 [M, 3] (numpy array, M <= N)
            如果return_details=True，还会返回:
            - desc_time: 描述符计算时间
            - cls_time: 分类时间
            - refine_time: 细化时间（如果refine=True）
        """
        if not STAR_EDGE_AVAILABLE:
            raise RuntimeError("STAR-Edge模块不可用")
        
        # 确保points是numpy array且是float64
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        points = points.astype(np.float64)
        
        if points.shape[1] != 3:
            raise ValueError(f"点云形状错误，期望[N, 3]，得到{points.shape}")
        
        N_original = points.shape[0]
        
        # ========== 临时降采样以加速STAR-Edge处理 ==========
        # 注意：这只是STAR-Edge内部的临时下采样，不影响用于渲染的原始点云密度
        # 边缘点会通过KD树匹配映射回原始点云
        if N_original > self.max_points:
            # 使用均匀随机采样
            downsample_ratio = self.max_points / N_original
            indices = np.random.choice(N_original, self.max_points, replace=False)
            indices = np.sort(indices)  # 保持顺序
            points_sampled = points[indices]
            if self.verbose:
                print(f"[STAR-Edge] 临时降采样: {N_original} -> {self.max_points} 点 (比例: {downsample_ratio:.2%})")
        else:
            points_sampled = points
            indices = np.arange(N_original)
        
        N = points_sampled.shape[0]
        if self.verbose:
            print(f"[STAR-Edge] 开始提取轮廓 - 处理点云数量: {N}")
        
        # ========== 步骤1: 计算局部球面曲线特征 ==========
        start_time = time.time()
        if self.verbose:
            print(f"[STAR-Edge] [步骤1/3] 计算局部球面曲线特征 (LocalSH)...")
        
        result = localsh_feature.ComLSHF_knn_upsample(
            points_sampled, self.bw, self.kk, self.sampleNum
        )
        Descs = result["Descs"]  # [N, 10]
        normals = result["normals"]  # [N, 3]
        neighboor = result["neighboor"]
        
        desc_time = time.time() - start_time
        if self.verbose:
            print(f"[STAR-Edge] [步骤1/3] 完成 - 耗时: {desc_time:.3f}秒")
            print(f"[STAR-Edge]        - 描述符形状: {Descs.shape}")
            print(f"[STAR-Edge]        - 法向量形状: {normals.shape}")
        
        # ========== 步骤2: 分类预测边缘点 ==========
        start_time = time.time()
        if self.verbose:
            print(f"[STAR-Edge] [步骤2/3] 使用分类网络预测边缘点...")
        
        Descs_tensor = torch.tensor(Descs, device=self.device).float()
        with torch.no_grad():
            pred = self.net(Descs_tensor)  # [N]
        
        edge_mask = (pred.cpu().numpy() > 0.5).astype(bool)
        num_edges = edge_mask.sum()
        edge_ratio = num_edges / N * 100
        
        cls_time = time.time() - start_time
        if self.verbose:
            print(f"[STAR-Edge] [步骤2/3] 完成 - 耗时: {cls_time:.3f}秒")
            print(f"[STAR-Edge]        - 边缘点数量: {num_edges}/{N} ({edge_ratio:.2f}%)")
        
        # ========== 步骤3: 边缘点细化（可选）==========
        refine_time = 0.0
        if refine:
            start_time = time.time()
            if self.verbose:
                print(f"[STAR-Edge] [步骤3/3] 边缘点细化 (NormalRefine)...")
            
            flag = edge_mask.astype(int)
            edge_points = normal_refine.EdgePointRefine(
                flag, neighboor, points_sampled, normals, self.mu
            )
            
            refine_time = time.time() - start_time
            if self.verbose:
                print(f"[STAR-Edge] [步骤3/3] 完成 - 耗时: {refine_time:.3f}秒")
                print(f"[STAR-Edge]        - 细化后边缘点数量: {len(edge_points)}")
        else:
            edge_points = points_sampled[edge_mask]
            if self.verbose:
                print(f"[STAR-Edge] 跳过细化步骤")
        
        total_time = desc_time + cls_time + refine_time
        if self.verbose:
            print(f"[STAR-Edge] ========== 轮廓提取完成 ==========")
            print(f"[STAR-Edge] 总耗时: {total_time:.3f}秒")
            print(f"[STAR-Edge] 边缘点比例: {num_edges}/{N} ({edge_ratio:.2f}%)")
            print(f"[STAR-Edge] ==================================")
        
        if return_details:
            return edge_mask, edge_points, {
                'desc_time': desc_time,
                'cls_time': cls_time,
                'refine_time': refine_time,
                'total_time': total_time,
                'num_edges': num_edges,
                'edge_ratio': edge_ratio
            }
        else:
            return edge_mask, edge_points


def extract_edges_simple(points, model_path=None, refine=True, verbose=True):
    """
    简化的边缘提取接口（一次性使用）
    
    Args:
        points: 点云 [N, 3] (numpy array)
        model_path: 模型路径（可选）
        refine: 是否细化
        verbose: 是否输出详细信息
    
    Returns:
        edge_mask: 边缘点标记 [N] (boolean array)
        edge_points: 边缘点云 [M, 3] (numpy array)
    """
    extractor = STAREdgeExtractor(model_path=model_path, verbose=verbose)
    return extractor.extract_edges(points, refine=refine)

