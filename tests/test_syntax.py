import logging
import sys

# 1. 创建 Logger 实例
# 通常，在应用程序中每个模块或组件会创建一个 Logger
logger = logging.getLogger(__name__)

# 默认级别是 WARNING，我们需要手动设置为 DEBUG 才能看到所有信息
logger.setLevel(logging.DEBUG)

# 2. 定义格式器 (Formatter)
# 定义日志输出的样式，包含时间、级别、模块名和消息
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# 3. 创建处理器 (Handler) - 控制台输出
# Console Handler 用于将日志信息输出到标准输出 (stdout)
console_handler = logging.StreamHandler(sys.stdout)
# 设置控制台只输出 INFO 级别及以上的信息
console_handler.setLevel(logging.INFO)
# 为控制台处理器应用格式
console_handler.setFormatter(formatter)

# 4. 创建处理器 (Handler) - 文件输出
# File Handler 用于将日志信息写入文件
file_handler = logging.FileHandler("app_activity.log")
# 设置文件输出所有 DEBUG 级别及以上的信息
file_handler.setLevel(logging.DEBUG)
# 为文件处理器应用格式
file_handler.setFormatter(formatter)

# 5. 将处理器添加到 Logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --- 应用程序的日志记录 ---

logger.debug("这是只有在开发模式下才能看到的最详细的调试信息。")
logger.info("程序启动成功，正在加载配置文件...")

try:
    result = 10 / 0
except ZeroDivisionError:
    # 推荐使用 logger.exception() 记录异常，它会自动包含完整的堆栈信息
    logger.exception("操作失败！尝试进行除零操作。")

logger.warning("外部服务连接超时，将使用本地缓存数据。")
logger.error("关键配置参数缺失，程序将以非最佳状态运行。")
