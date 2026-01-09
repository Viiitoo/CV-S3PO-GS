import rich
from rich.console import Console
from rich.text import Text

console = Console()

_log_styles = {
    "S3PO-GS": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
    "Deform": "bold cyan",
    "Init": "bold yellow",
    "Keyframe": "bold blue",
    "BA": "bold magenta",
    "Densify": "bold yellow",
    "WARNING": "bold red",
    "INFO": "bold blue",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="S3PO-GS"):
    """主日志输出函数，使用颜色标记不同的标签"""
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)


def format_percentage(value, decimals=1):
    """格式化百分比输出，带颜色"""
    if value >= 0.8:
        color = "green"
    elif value >= 0.5:
        color = "yellow"
    elif value >= 0.2:
        color = "orange1"
    else:
        color = "red"
    return f"[{color}]{value*100:.{decimals}f}%[/{color}]"


def format_number(value, color=None):
    """格式化数字输出，可选颜色"""
    if color:
        return f"[{color}]{value}[/{color}]"
    return str(value)


def print_status(message, status="info"):
    """打印状态信息，带颜色标记"""
    status_colors = {
        "info": "blue",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "highlight": "cyan",
    }
    color = status_colors.get(status, "blue")
    console.print(f"[{color}]{message}[/{color}]")
