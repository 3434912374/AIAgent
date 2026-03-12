import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def run_multi_agent_collaboration(api_key: str, task_query: str):
    """
    运行基于最新版 AutoGen (0.10.x) 的异步 Multi-Agent 协作
    """
    print("\n" + "="*50)
    print("🚀 AutoGen (新版异步架构) 协作开始...")
    print("="*50)

    model_client = OpenAIChatCompletionClient(
            model="deepseek-chat",  # 💡 建议使用 DeepSeek 官方建议的模型名
            api_key=api_key,
            # 💡 核心修改点 1: 提供模型基本信息，让 AutoGen 不再报错
            model_info=ModelInfo(
                vision=False, 
                function_calling=True, 
                json_output=True, 
                family="unknown"
            ),
            # 💡 核心修改点 2: 填入 DeepSeek 的 API 地址（如果你用的是 DeepSeek 官方 Key）
            base_url="https://api.deepseek.com/v1", 
            temperature=0.7,
        )

    # 变化2：统一 Agent 类型。废弃了原本鸡肋的 UserProxyAgent，全员皆 Assistant
    researcher = AssistantAgent(
        name="Researcher",
        system_message="你是一个专业的AI行业研究员。你的任务是分析需求，提出核心的知识点和研究框架。",
        model_client=model_client,
    )

    writer = AssistantAgent(
        name="Writer",
        system_message="你是一个技术作家。你根据 Researcher 提供的框架，撰写一篇结构严谨、通俗易懂的短文。完成后务必在结尾输出 'TERMINATE'。",
        model_client=model_client,
    )

    # 变化3：显式定义终止条件 (条件组合)
    # 只要有人说出 TERMINATE，或者总对话轮数达到10轮，立刻停止，防止死循环无限消耗 Token。
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=10)

    # 变化4：引入 Team (团队) 概念
    # SelectorGroupChat 会利用 LLM 自动评估当前上下文，动态选择下一个最适合发言的 Agent
    team = SelectorGroupChat(
        participants=[researcher, writer],
        model_client=model_client,
        termination_condition=termination,
    )

    # 5. 启动异步对话，并使用官方提供的 Console UI 优雅地流式输出
    await Console(team.run_stream(task=task_query))


# 如果你想单独测试这个文件，可以加上这部分
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    API_KEY = os.getenv("OPENAI_API_KEY", "你的API_KEY")
    query = "请研究一下目前大模型 Tool Calling 的核心原理，并写一篇约300字的科普文章。"
    
    # 因为新版是纯异步架构，必须用 asyncio.run 来启动
    asyncio.run(run_multi_agent_collaboration(API_KEY, query))