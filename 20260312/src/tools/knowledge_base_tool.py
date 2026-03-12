from pydantic import BaseModel, Field
from langchain_core.tools import tool
from src.memory.vector_store import KnowledgeBaseManager

# 1. 实例化管理器并获取检索器
kb_manager = KnowledgeBaseManager()
retriever = kb_manager.get_retriever()

# 2. 定义输入格式规范（让大模型知道该传什么参数进来）
class KnowledgeBaseInput(BaseModel):
    query: str = Field(description="需要在知识库中搜索的关键词或完整问题")

@tool("knowledge_base_tool", args_schema=KnowledgeBaseInput)
def knowledge_base_tool(query: str) -> str:
    """当用户询问公司的通用政策、退换货规则、会员制度、物流配送、产品参数等静态知识时，必须使用此工具。不要用你的常识回答，必须检索知识库。"""
    
    # 调用底层检索器获取相关的文档块
    docs = retriever.invoke(query)
    
    if not docs:
        return "知识库中没有找到相关信息，请建议用户转人工客服。"
    
    # 将检索到的多个文档片段拼接成一段长文本，返回给大模型作为“观察结果 (Observation)”
    result_text = "\n\n".join([f"参考资料 {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    return result_text