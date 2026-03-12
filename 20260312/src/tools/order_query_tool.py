import asyncio
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field

class OrderQueryInput(BaseModel):
    """订单查询输入参数"""
    order_id: str = Field(description="需要查询的订单号，格式通常为 'ORD-' 开头")

    #指数退避重试机制：最大重试3次，每次邓丹时间指数增加
    @retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1, min=2, max=10))
    async def query_order(self,order_id:str)->str:
        """查询订单详情"""
        # 模拟异步查询订单详情的过程
        await asyncio.sleep(1)  # 模拟网络延迟
        if order_id.startswith("ORD-"):
            return f"订单号 {order_id} 的详情信息：\n- 状态：已发货\n- 商品：商品A\n- 数量：2\n- 单价：100元\n- 总价：200元"
        else:
            raise ValueError("无效的订单号格式")
    
@tool("order_query_tool",args_schema=OrderQueryInput)
async def order_query_tool(order_id:str)->str:
    """当用户询问其订单的状态、物流或详情时，使用此工具。"""
    try:
        result=await mock_api_call(order_id)
        return f"订单 {order_id} 当前状态为：{result['status']}，物流公司：{result['express_company']}。"
    except Exception as e:
        return f"抱歉，查询订单 {order_id} 时系统繁忙，请稍后再试。错误详情: {str(e)}"
    