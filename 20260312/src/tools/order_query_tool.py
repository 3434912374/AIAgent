import asyncio
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field

class OrderQueryInput(BaseModel):
    """订单查询输入参数"""
    order_id: str = Field(description="需要查询的订单号，格式通常为 'ORD-' 开头")

# 1. 修正：删掉 self，因为它不是类成员函数
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def query_order_api(order_id: str):
    """模拟异步查询订单详情的过程"""
    await asyncio.sleep(1)  # 模拟网络延迟
    
    # 模拟订单系统逻辑
    if order_id == "ORD-2026":
        return {"status": "已发货", "express_company": "顺丰速运"}
    elif order_id.startswith("ORD-"):
        return {"status": "配货中", "express_company": "待分配"}
    else:
        # 触发重试逻辑的异常
        raise ValueError("无效的订单号格式或订单不存在")

@tool("order_query_tool", args_schema=OrderQueryInput)
async def order_query_tool(order_id: str) -> str:
    """当用户询问其订单的状态、物流或详情时，使用此工具。"""
    try:
        # 2. 修正：调用上面定义好的 query_order_api
        result = await query_order_api(order_id)
        return f"订单 {order_id} 当前状态为：{result['status']}，物流公司：{result['express_company']}。"
    except Exception as e:
        # 如果重试3次后依然失败，返回友好提示
        return f"抱歉，查询订单 {order_id} 时系统繁忙或单号不存在，请核对后再试。错误详情: {str(e)}"