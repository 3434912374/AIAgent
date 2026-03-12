import asyncio
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.messages import content
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from src.workflows.customer_service_workflow import CustomerServiceWorkflow


async def main():
    # 性能优化：启用LLM的全局SQLite缓存，想通的问题直接返回，节约时间和成本
    set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))
     # 记忆优化：使用MemorySaver实现多轮对话的上下文管理
    checkpointer = MemorySaver()
    # 实例化工作流引擎
    workflow = CustomerServiceWorkflow(checkpointer=checkpointer)
    app = workflow.graph
    # 线程 ID 用于区分不同用户的会话，实现真正的多轮隔离
    config = {"configurable": {"thread_id": "user_123"}}

    print("智能客服系统启动 (输入 'quit' 退出)")
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == "quit":
            print("智能客服系统关闭")
            break

       #异步流执行状态机
        async for event in app.astream({"messages":[HumanMessage(content=user_input)]},config=config):
           for node_name,node_data in event.items():
             if "messages" in node_data and node_name in ["agent", "human_escalation"]:
                     last_msg = node_data["messages"][-1]
                     if not last_msg.tool_calls: # 如果不是工具调用请求，则是最终文本
                         print(f"客服 [{node_name}]: {last_msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
