from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain.agents import create_agent

qwen14bllm = ChatOpenAI(
    model="qwen-14b-awq",
    openai_api_key="none",
    openai_api_base="http://192.168.2.8:8000/v1",
    temperature=0.7,
)

# 智能代理 (Agent)
@tool
def calculate_area(radius: float):
    """计算圆的面积。当你需要根据半径计算面积时，请调用此工具。"""
    import math

    return math.pi * radius**2


tools = [calculate_area]


def run_agent_task(query):
    agent_executor = create_agent(qwen14bllm, tools)
    # 2. 执行任务 (注意：LangGraph 的输入格式变成了标准的 messages 列表)
    result = agent_executor.invoke({"messages": [("user", query)]})
    # 3. 提取对话历史中的最后一句话作为最终结果
    return result["messages"][-1].content


# 简单链
def run_simple_chain(user_input):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个起名专家，擅长起具有古风意境的名字。"),
            ("human", "请为一家{store_type}起3个名字。"),
        ]
    )
    chain = prompt | qwen14bllm | StrOutputParser()
    result = chain.invoke({"store_type": user_input})
    return result


# 顺序链
def run_sequence_chain(topic):
    # 步骤 1: 根据主题写出 3 个核心关键词
    prompt1 = ChatPromptTemplate.from_template(
        "关于'{topic}'，请给出3个最关键的行业术语。"
    )
    # 步骤 2: 根据关键词写一段科普介绍
    prompt2 = ChatPromptTemplate.from_template(
        "请根据以下术语：{terms}，写一段通俗易懂的科普。"
    )
    # 组合流水线
    chain = (
        {"terms": prompt1 | qwen14bllm | StrOutputParser()}
        | prompt2
        | qwen14bllm
        | StrOutputParser()
    )
    result = chain.invoke({"topic": topic})
    return result



