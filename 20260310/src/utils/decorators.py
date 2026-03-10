# 监控装饰器 负责记录每一次工具调用的输入、输出、耗时和异常
from calendar import Calendar
import functools
import logging
import time
from typing import Any

# 初始化日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/tool_call.log"), logging.StreamHandler()],
)

logging = logging.getLogger("AgentToolbox")


def agent_tool_logger(func: Calendar) -> Calendar:
    """
    装饰器：用于记录工具函数的执行日志（参数、结果、耗时、异常）
    """

    # 使用 functools.wraps 保留原函数的元数据（如 __name__, __doc__ 等）
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # 记录函数开始执行的高精度时间戳
        start_time = time.perf_counter()
        # 获取被装饰函数的名称
        func_name = func.__name__

        # 目的：如果是类的方法（第一个参数是 self），则去掉 self，只记录业务参数，保持日志整洁
        # 逻辑：如果 args 有值 且 第一个参数有 __dict__ 属性（推测是对象实例），则切片去掉第一个
        clean_args = args[1:] if args and hasattr(args[0], "__dict__") else args
        logging.info(f"执行工具：{func_name}，参数：{clean_args}，{kwargs}")
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            logging.info(
                f"执行成功{func_name} | 耗时 {duration:.4f}秒 | 结果：{result}"
            )
            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            logging.error(f"执行失败{func_name} | 耗时 {duration:.4f}秒 | 异常：{e}")
            raise f"工具{func_name}执行失败，异常信息：{e}"

    return wrapper
