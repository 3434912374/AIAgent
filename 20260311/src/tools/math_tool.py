from langchain.tools import tool

@tool
def calculator_tool(expression:str)->str:
    """用于进行简单的数学计算"""
    try:
        result=eval(expression)
        return f"计算结果是：{result}"
    except Exception as e:
        return f"计算错误：{e}"