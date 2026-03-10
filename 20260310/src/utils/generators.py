from typing import Any, Generator, Iterable


def batch_data_generator(data: Iterable) -> Generator[Any, None, None]:
    """
    通用生成器：将大数据集转化为流式，模拟 Agent 的增量处理
    """
    for item in data:
        yield item
