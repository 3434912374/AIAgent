from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class UserIntent(BaseModel):
    intent_type:str=Field(description="意图类型，可选值: 'qa'(知识库问答), 'order'(订单查询), 'complain'(投诉转人工), 'other'(日常闲聊)")
    requires_tools:bool=Field(description="是否需要调用工具才能回答")
    summary:str=Field(description="用户意图的简洁描述")

def get_intent_chain(llm:ChatOpenAI):
     """创建意图识别链"""
     prompt = ChatPromptTemplate.from_messages([
         ("system", "你是一个高级意图识别引擎。请分析用户的最后一条消息，并判断其意图。"),
         ("placeholder", "{messages}")
     ])
     # 强制指定 method="function_calling" 来兼容 DeepSeek
     return prompt | llm.with_structured_output(UserIntent, method="function_calling")
