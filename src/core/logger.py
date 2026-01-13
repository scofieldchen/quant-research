import sys
from pathlib import Path

from loguru import logger as loguru_logger

# 全局日志配置：共享日志文件
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
GLOBAL_LOG_FILE = LOG_DIR / "app.log"

# 初始化全局处理器
loguru_logger.remove()
loguru_logger.add(
    GLOBAL_LOG_FILE,
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[name]} | {message}",
)
loguru_logger.add(
    sys.stdout,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[name]} | {message}",
    colorize=True,
)


def get_logger(name: str = "app", log_file: str = None):
    """
    返回一个绑定名称的 loguru logger 对象，默认使用全局日志文件。
    如果提供 log_file，则使用指定的文件（用于特定管道的隔离日志）。

    Args:
        name (str): logger 的名称，用于标识日志来源（例如 "sth_mvrv"）。
        log_file (str, optional): 可选的日志文件路径，支持 loguru 格式化。如果提供，将添加额外处理器。

    Returns:
        loguru.Logger: 配置好的 logger 对象。
    """
    logger = loguru_logger.bind(name=name)
    if log_file:
        # 为特定管道添加额外文件处理器（如果需要隔离）
        logger.add(
            log_file,
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[name]} | {message}",
        )
    return logger
