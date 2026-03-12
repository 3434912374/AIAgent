import os
import asyncio
from dotenv import load_dotenv
from src.agents.tool_agent import create_advanced_agent
from src.autogen_demo.group_chat import run_multi_agent_collaboration

load_dotenv()
API_KEY=os.getenv("OPENAI_API_KEY")



def main():
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY 未配置")

    print("请选择要运行的模块:")
    print("1. LangChain 单体工具 Agent (天气/搜索/计算)")
    print("2. AutoGen 多 Agent 协作 (研究员与作家)")
    choice = input("请输入 1 或 2: ")

    if choice == '1':
        agent_app = create_advanced_agent(API_KEY)
        query = "请帮我查一下今天旧金山的天气，并计算一下 25的平方加上100是多少，另外搜一下今天科技界的最新大新闻。"
        
        # 💡 把系统设定当做 "system" 角色消息，直接喂给状态机，100% 兼容所有版本！
        system_prompt = "你是一个强大的AI助手，拥有天气查询、网络搜索和数学计算的能力。请根据用户需求，合理选择工具进行解答。"
        
        print(f"\n用户提问: {query}\n")
        response = agent_app.invoke({"messages": [
            ("system", system_prompt),
            ("user", query)
        ]})
        
        final_answer = response["messages"][-1].content
        print(f"\n最终回答: {final_answer}")

    elif(choice=='2'):
        query = "请研究一下目前大模型 Tool Calling (函数调用) 的核心原理，并写一篇约300字的科普文章。"
        asyncio.run(run_multi_agent_collaboration(API_KEY, query))
    else:
        print("无效输入。")

if __name__ == "__main__":
    main()