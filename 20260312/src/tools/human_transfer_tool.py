from pydantic import BaseModel, Field
from langchain_core.tools import tool

class HumanTransferInput(BaseModel):
    reason: str = Field(description="转接人工客服的原因")

@tool("human_transfer_tool", args_schema=HumanTransferInput)
def human_transfer_tool(self, reason: str) -> str:
    """当用户明确要求人工服务，或者遇到极其复杂、情感激动的问题时，使用此工具。"""
    # 在真实场景中，这里会调用 websocket 或 webhook 通知人工坐席系统
    return f"__TRANSFER_HUMAN__:{reason}" # 返回特定标识符供工作流引擎识别
