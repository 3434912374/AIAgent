# 20260312 客服智能体项目

# 项目依赖
python -m pip install -U langchain langchain-core langchain-openai langchain-community langgraph faiss-cpu pydantic tenacity

# 项目描述
langgraph: 你的大脑皮层。它负责执行 StateGraph，管理 recognize_intent -> agent -> tools 的流转，并记录多轮对话记忆（Checkpointer）。

langchain-openai: 你的语言中枢。虽然你用的是 DeepSeek，但因为 DeepSeek 兼容 OpenAI 的 API 格式，所以我们用它底层的 ChatOpenAI 类来发起网络请求。

pydantic: 你的安检员。在 OrderQueryInput 和意图识别的 UserIntent 中，它强制大模型输出格式正确的数据，如果没有它，Agent 极容易产生格式幻觉。

tenacity: 你的防弹衣。在 order_query_tool 中，我们用了 @retry，它确保了当外部订单系统崩溃或超时时，Agent 不会跟着崩溃，而是优雅地重试。

faiss-cpu: 你的外挂记忆库。用于把文本变成向量（Embeddings）存起来，实现 RAG（检索增强生成）知识库问答。

langchain-core / langchain-community: 你的基础设施。提供了像 HumanMessage、ToolNode 以及我们用来做性能优化的 SQLiteCache。
