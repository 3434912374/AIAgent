from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from src.tools.langchain_adapter import tools


def start_qwen_agent():
    llm = ChatOpenAI(
        model="qwen-14b-awq",
        openai_api_key="none",
        temperature=0, 
        max_tokens=512,
        openai_api_base="http://192.168.2.8:8000/v1",
    )
    agent = create_agent(
        llm, tools=tools, system_prompt="你是一个智能助手，请调用工具回答问题。"
    )

    query = "帮我计算一个半径为 5.5 的圆面积，并分析这段话的关键词：'Python is great for AI, especially for building AI Agents.'"

    result = agent.invoke({"messages": [("human", query)]})

    print(result["messages"][-1].content)

if __name__ == "__main__":
    start_qwen_agent()

# from src.tools.calculator import CalculatorTool
# from src.tools.file_tool import FileTool
# from src.tools.text_tool import TextTool
# from src.utils.generators import batch_data_generator


# def run_agent_workflow():
#     """运行Agent工作流程"""
#     # 初始化Agent
#     calc = CalculatorTool()
#     writer = FileTool()
#     text_processor = TextTool()

#     print("--- 1.批量数据流式处理示例 ---")
#     radii = [1.5, 3.0, 1.0, 5.2]  # 故意包含负数
#     gen = batch_data_generator(radii)

#     results = []
#     for r in gen:
#         area = calc.calculate_circle_area(r)
#         results.append(f"半径{r}的圆面积为{area:.2f}")

#     print("--- 2. 文本清洗与持久化 (数据结构+OOP) ---")
#     raw_text = "Agent is smart, Agent is fast, Agent is efficient."
#     keywords = text_processor.extract_unique_keywords(raw_text)
#     final_report = f"提取到的关键词: {keywords}\n" + "\n".join(results)

#     print("--- 3. 词频统计 ---")
#     freq_stats = text_processor.word_frequency_stats(raw_text)
#     print("词频统计结果:", freq_stats)

#     # 保存结果
#     status = writer.save_analysis_result("daily_report.txt", final_report)
#     print("保存分析结果状态:", status)


# if __name__ == "__main__":
#     run_agent_workflow()
