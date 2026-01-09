#!/bin/bash
# 生成 PDF 的脚本

INPUT_MD="/home/sjw/data0/lsx/S3PO_baseline/docs/2025.1.1形变处理工作汇报.md"
OUTPUT_PDF="/home/sjw/data0/lsx/S3PO_baseline/docs/2025.1.1形变处理工作汇报.pdf"
OUTPUT_HTML="/home/sjw/data0/lsx/S3PO_baseline/docs/2025.1.1形变处理工作汇报.html"

echo "尝试生成 PDF..."

# 方法 1: 尝试使用 pandoc
if command -v pandoc &> /dev/null; then
    echo "使用 pandoc 生成 PDF..."
    pandoc "$INPUT_MD" -o "$OUTPUT_PDF" --pdf-engine=xelatex -V CJKmainfont='Microsoft YaHei' 2>/dev/null
    if [ -f "$OUTPUT_PDF" ]; then
        echo "✓ PDF 已生成: $OUTPUT_PDF"
        exit 0
    fi
fi

# 方法 2: 尝试使用 wkhtmltopdf
if command -v wkhtmltopdf &> /dev/null; then
    echo "使用 wkhtmltopdf 生成 PDF..."
    if [ -f "$OUTPUT_HTML" ]; then
        wkhtmltopdf "$OUTPUT_HTML" "$OUTPUT_PDF" 2>/dev/null
        if [ -f "$OUTPUT_PDF" ]; then
            echo "✓ PDF 已生成: $OUTPUT_PDF"
            exit 0
        fi
    fi
fi

# 方法 3: 尝试使用 markdown-pdf (npm)
if command -v markdown-pdf &> /dev/null; then
    echo "使用 markdown-pdf 生成 PDF..."
    markdown-pdf "$INPUT_MD" -o "$OUTPUT_PDF" 2>/dev/null
    if [ -f "$OUTPUT_PDF" ]; then
        echo "✓ PDF 已生成: $OUTPUT_PDF"
        exit 0
    fi
fi

echo "无法自动生成 PDF，请使用浏览器打开 HTML 文件并打印为 PDF:"
echo "  $OUTPUT_HTML"
exit 1


