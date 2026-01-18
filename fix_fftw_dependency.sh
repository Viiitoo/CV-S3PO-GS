#!/bin/bash
# 修复 LocalSH.so 的 fftw 依赖

set -e

SO_FILE="/workspace/STAR-Edge/pre_process/LocalSH.cpython-38-x86_64-linux-gnu.so"
FFTW_LIB="/usr/lib/x86_64-linux-gnu/libfftw3.so.3.5.8"

echo "检查 patchelf..."
if ! command -v patchelf &> /dev/null; then
    echo "安装 patchelf..."
    apt-get update
    apt-get install -y patchelf
fi

echo "检查符号链接..."
if [ ! -L "/usr/lib/x86_64-linux-gnu/libfftw3.so.3.6.9" ]; then
    echo "创建符号链接..."
    ln -sf "$FFTW_LIB" /usr/lib/x86_64-linux-gnu/libfftw3.so.3.6.9
    ldconfig
fi

echo "修改 .so 文件的依赖..."
# 将依赖从 libfftw3.so.3.6.9 改为 libfftw3.so.3
patchelf --replace-needed libfftw3.so.3.6.9 libfftw3.so.3 "$SO_FILE" || {
    echo "patchelf 修改失败，尝试其他方法..."
    # 如果 patchelf 失败，至少确保符号链接存在
    echo "请确保符号链接存在: /usr/lib/x86_64-linux-gnu/libfftw3.so.3.6.9"
}

echo "验证修改..."
readelf -d "$SO_FILE" | grep -i fftw

echo "完成！"


