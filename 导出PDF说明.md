# 将工作汇报导出为PDF的几种方法

## 方法1：使用已生成的HTML文件（推荐，最简单）

HTML文件已经生成：`docs/形变处理工作汇报.html`

1. 在浏览器中打开该HTML文件
2. 使用浏览器的打印功能（Ctrl+P 或 Cmd+P）
3. 选择"另存为PDF"
4. 保存即可

## 方法2：安装pandoc后转换

```bash
# 安装pandoc
sudo apt-get update
sudo apt-get install -y pandoc texlive-latex-base texlive-fonts-recommended

# 转换为PDF
pandoc docs/形变处理工作汇报.md -o docs/形变处理工作汇报.pdf --pdf-engine=xelatex -V CJKmainfont="Microsoft YaHei"
```

## 方法3：使用Python库（需要安装依赖）

```bash
# 安装依赖
pip install markdown weasyprint

# 运行转换脚本
python3 convert_to_pdf.py
```

## 方法4：使用在线工具

1. 将Markdown文件上传到在线转换工具（如：https://www.markdowntopdf.com/）
2. 下载生成的PDF

## 当前状态

- ✅ HTML文件已生成：`docs/形变处理工作汇报.html`
- ❌ PDF文件需要进一步转换

建议使用方法1（浏览器打印），这是最简单且不需要额外安装软件的方法。


