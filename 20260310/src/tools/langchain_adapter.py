from langchain.tools import tool
from src.tools.calculator import CalculatorTool
from src.tools.file_tool import FileTool
from src.tools.text_tool import TextTool

# 实例化你的 OOP 工具类
calc_instance=CalculatorTool()
file_instance=FileTool()
text_instance=TextTool()

@tool
def calculate_circle_area(radius: float) -> str:
    """计算圆的面积。当你需要处理几何问题或计算圆形物体的大小时使用此工具。
    参数 radius 必须是正数。"""
    try:
        area=calc_instance.calculate_circle_area(radius)
        return f"半径{radius}的圆面积为{area:.2f}"
    except ValueError as e:
        return str(e)

@tool
def process_text_keywords(text: str) -> str:
    """
    从给定的文本中提取唯一的关键词并进行词频统计。
    适用于文本分析、摘要提取或关键词搜索任务。
    """
    keywords = text_instance.extract_unique_keywords(text)
    stats = text_instance.word_frequency_stats(text)
    return f"关键词: {keywords}\n词频统计: {stats}"

tools=[calculate_circle_area,process_text_keywords]
