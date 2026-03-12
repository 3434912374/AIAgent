from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from src.tools.weather_tool import WeatherTool
from src.tools.search_tool import get_search_tool
from src.tools.math_tool import calculator_tool

def create_advanced_agent(api_key: str):
    llm = ChatOpenAI(
        model="deepseek-chat",  
        temperature=0, 
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"  
    )
    tools = [WeatherTool(), get_search_tool(), calculator_tool]
    
    # 💡 终极奥义：去掉所有花里胡哨的 modifier 参数，只传模型和工具
    agent_app = create_react_agent(llm, tools)
    return agent_app