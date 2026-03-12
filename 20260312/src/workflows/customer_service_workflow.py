import operator
from typing import Annotated, Self, Sequence, TypedDict
from langchain_core.messages import BaseMessage,AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode
#引入写好的模块
from src.chains.intent_recognition_chain import get_intent_chain
from src.tools.order_query_tool import order_query_tool
from src.tools.human_transfer_tool import human_transfer_tool

# 1. 定义图的全局状态 (State)
class AgentState(TypedDict):
    """定义客服代理的全局状态"""
    messages:Annotated[Sequence[BaseMessage],operator.add]#消息会自动追加
    intent_type:str
    needs_human:bool

class CustomerServiceWorkflow:
    def __init__(self,checkpointer=None):
        self.llm = ChatOpenAI(
            model="deepseek-chat",  # 换成 DeepSeek 的模型名
            temperature=0, 
            api_key="sk-c45174ee28f847f8bf59669c79cc7383",
            base_url="https://api.deepseek.com/v1"
        )
        self.tools=[order_query_tool,human_transfer_tool]
        self.llm_with_tools=self.llm.bind_tools(self.tools)

        #初始化意图链
        self.intent_chain=get_intent_chain(self.llm)
        self.graph=self._build_graph(checkpointer)

    def _build_graph(self,checkpointer=None):
        """构建客服代理的状态图"""
        workflow=StateGraph(AgentState)
        workflow.add_node("recognize_intent",self.recognize_intent_node)# 意图识别节点
        workflow.add_node("agent",self.agent_node)# Agent核心思考与回复节点
        workflow.add_node("tools",ToolNode(self.tools))# LangGraph内置工具执行节点
        workflow.add_node("human_escalation",self.human_escalation_node)# 人工客服转接节点

        #设置入口点
        workflow.set_entry_point("recognize_intent")

        #意图识别后的条件路由
        workflow.add_conditional_edges(
            "recognize_intent",
            self.route_based_on_intent,
            {
                "to_agent": "agent",
                "to_human": "human_escalation"
            }
        )

        # Agent决定是否调用工具的条件路由
        workflow.add_conditional_edges(
            "agent",
            self.should_continue_or_tool,
            {
                "continue": END,
                "call_tool": "tools"
            }
        )
        

        # 工具执行完毕后，返回Agent节点生成最终回复
        workflow.add_edge("tools", "agent")
        workflow.add_edge("human_escalation", END)

        return workflow.compile(checkpointer=checkpointer)


    # --- 节点逻辑实现 ---
    def recognize_intent_node(self,state:AgentState):
        """意图识别节点"""
        intent = self.intent_chain.invoke({"messages":state["messages"]})
        needs_human=(intent.intent_type=="complain")
        return {"intent_type":intent.intent_type,"needs_human":needs_human}
        
    def route_based_on_intent(self,state:AgentState):
        """路由函数：根据意图决定下一步"""
        if state["needs_human"]:
            return "to_human"
        return "to_agent"

    def agent_node(self,state:AgentState):
        """Agent核心思考与回复节点"""
        # 带入上下文调用大模型
        response=self.llm_with_tools.invoke(state["messages"])
        return {"messages":[response]}
    
    def should_continue_or_tool(self,state:AgentState):
        """判断是否继续思考或调用工具"""
        last_message=state["messages"][-1]
        if last_message.tool_calls:
            return "call_tool"
        return "continue"

    def human_escalation_node(self,state:AgentState):
        """人工客服转接节点"""
        msg = AIMessage(content="检测到您的问题较复杂，正在为您转接高级人工客服，请稍候...")
        return {"messages": [msg], "needs_human": True}
 