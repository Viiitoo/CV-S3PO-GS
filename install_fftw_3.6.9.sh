#!/bin/bash
# 安装 FFTW 3.6.9 的脚本

set -e

echo "开始安装 FFTW 3.6.9..."

# 检查是否已安装
if [ -f "/usr/local/lib/libfftw3.so.3.6.9" ]; then
    echo "FFTW 3.6.9 已安装，跳过安装步骤"
    exit 0
fi

# 创建工作目录
WORK_DIR="/tmp/fftw_install"
mkdir -p $WORK_DIR
cd $WORK_DIR

# 下载 FFTW 3.6.9 源码
echo "下载 FFTW 3.6.9 源码..."
FFTW_VERSION="3.6.9"
FFTW_URL="http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"

if [ ! -f "fftw-${FFTW_VERSION}.tar.gz" ]; then
    wget $FFTW_URL || {
        echo "下载失败，尝试备用链接..."
        # 如果官方链接失败，可以尝试其他镜像
        wget "https://github.com/FFTW/fftw3/archive/refs/tags/${FFTW_VERSION}.tar.gz" -O "fftw-${FFTW_VERSION}.tar.gz" || {
            echo "所有下载链接都失败，请手动下载 fftw-${FFTW_VERSION}.tar.gz"
            exit 1
        }
    }
fi

# 解压
echo "解压源码..."
tar -xzf fftw-${FFTW_VERSION}.tar.gz
cd fftw-${FFTW_VERSION}

# 配置和编译
echo "配置 FFTW..."
./configure --prefix=/usr/local --enable-shared --enable-threads

echo "编译 FFTW（这可能需要几分钟）..."
make -j$(nproc)

echo "安装 FFTW..."
sudo make install

# 更新库路径
echo "更新库路径..."
sudo ldconfig

# 验证安装
if [ -f "/usr/local/lib/libfftw3.so.3.6.9" ]; then
    echo "✓ FFTW 3.6.9 安装成功！"
    echo "库文件位置: /usr/local/lib/libfftw3.so.3.6.9"
else
    echo "✗ 安装可能失败，请检查错误信息"
    exit 1
fi

# 清理
cd /
rm -rf $WORK_DIR

echo "安装完成！"


