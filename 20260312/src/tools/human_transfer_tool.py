from pydantic import BaseModel, Field
from langchain_core.tools import tool


class HumanTransferInput(BaseModel):
    reason: str = Field(description="转接人工客服的具体原因，例如：用户要求退货、订单查询失败等")


@tool("human_transfer_tool", args_schema=HumanTransferInput)
def human_transfer_tool(reason: str) -> str:
    """当用户明确要求人工服务，或者遇到极其复杂、投诉、退货申请等问题时，使用此工具。"""
    
    print(f"\n[🔧 工具执行] 正在触发转人工流程，原因: {reason}")
    
    # 返回特定的标识符，我们在 LangGraph 的节点逻辑中可以捕获这个标识
    return f"__TRANSFER_HUMAN_ACTION__: {reason}"