# AI Agent 开发与工具集成实践

## 项目简介
本项目是 AI Agent 学习计划的第3天产出。展示了如何使用 LangChain 和 AutoGen 框架构建具有外部工具调用能力和多 Agent 协作能力的智能系统。

## 核心特性
* **架构分层**：严格区分 Tools（工具层）、Agents（逻辑层）和 Multi-Agent（协作层）。
* **高级工具调用**：不使用简单的字符串解析，而是采用 OpenAI Native Tool Calling 结合 Pydantic 进行严格的参数校验。
* **多模型协作**：基于 AutoGen 实现了 User, Researcher, Writer 的 GroupChat 协作机制。
* **多工具集成**：集成了 DuckDuckGo 网络搜索、模拟天气 API、以及数学表达式计算器。

## 快速启动
1. 克隆本项目。
2. 运行 `pip install -r requirements.txt` 安装依赖。
3. 在根目录创建 `.env` 文件，填入：`OPENAI_API_KEY=sk-xxxx`
4. 运行 `python main.py`，根据控制台提示体验功能。

## 项目结构
```
├── src/
│   ├── autogen_demo/
│   │   ├── group_chat.py  # AutoGen 多 Agent 协作层
│   ├── tools/
│   │   ├── search_tool.py  # 网络搜索工具
│   │   ├── weather_tool.py  # 天气查询工具
│   │   ├── math_tool.py  # 数学计算工具
├── main.py  # 主程序入口
├── requirements.txt  # 项目依赖
├── .env  # 环境变量配置文件
```

## 安装依赖
pip install python-dotenv requests pydantic langchain langchain-openai langgraph duckduckgo-search autogen-agentchat autogen-ext