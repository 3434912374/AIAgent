import math
from src.utils.decorators import agent_tool_logger


class CalculatorTool:
    """高级数学计算工具，支持 Agent 进行几何与统计计算"""

    @agent_tool_logger
    def calculate_circle_area(self, radius: float) -> float:
        """计算圆的面积"""
        if radius <= 0:
            raise ValueError("半径必须大于0")
        return math.pi * radius**2

    @agent_tool_logger
    def calulate_compound_interest(
        self, principal: float, rate: float, time: float, n: int = 12
    ) -> float:
        """计算复利 公式：A = P(1 + r/n)^(nt) principal:本金,rate:年利率,time:时间,n: 每年计息次数"""
        if principal <= 0 or rate <= 0 or time <= 0 or n <= 0:
            raise ValueError("参数必须大于0")
        return principal * (1 + rate / n) ** (n * time)
