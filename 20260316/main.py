import yaml
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from rag_engine.embeddings import DenseEmbedder
from rag_engine.retrievers import DualRetriever
from rag_engine.fusion import reciprocal_rank_fusion
from rag_engine.reranker import DeepReranker
from rl_scheduler.ppo_agent import PPOScheduler
from workflow.telemetry import setup_telemetry
from workflow.dag_engine import DAGWorkflow

# --- 新增：查询重写（Query Rewrite）引擎 ---
def rewrite_query(original_query, api_key):
    """
    检索前置拦截器（收紧版）：严格限制输出长度，防止大模型词汇爆炸。
    """
    print(f"  [重写引擎] 正在请 DeepSeek 提取核心检索词...")
    llm = ChatOpenAI(
        model="deepseek-chat",  
        temperature=0,  # 必须设为 0，让它绝对冷静，不发散
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"  
    )
    
    # 【核心修改点】：给 Prompt 加上极度严格的限制
    prompt = f"""你是一个搜索引擎优化专家。
请从以下用户的口语化提问中，提取最重要的 3 个技术关键词。如果涉及到寻找特定工具，你可以直接补充 1 个最著名的开源工具名。

用户的原始提问：“{original_query}”

【严格要求】：
1. 你的输出只能包含 3-4 个词语。
2. 词语之间用空格隔开。
3. 绝对不要输出任何标点符号、解释性文字或其他废话。
"""
    messages = [
        SystemMessage(content="你是一个严格的关键词提取器。输出不能超过 4 个词。"),
        HumanMessage(content=prompt)
    ]
    
    expanded_query = llm.invoke(messages).content.strip()
    print(f"  [重写引擎] 原始问题: {original_query}")
    print(f"  [重写引擎] 精炼检索词: {expanded_query}")
    return expanded_query
# ----------------------------------------

def generate_answer(query, retrieved_docs, api_key):
    context = "\n".join([f"片段 {i+1}: {doc['text']}" for i, doc in enumerate(retrieved_docs)])
    llm = ChatOpenAI(
        model="deepseek-chat",  
        temperature=0, 
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"  
    )
    prompt = f"""你是一个专业的企业级 AI 助手。请严格基于以下【检索到的参考知识】来回答用户的问题。
如果知识库中没有相关内容，请直接回答“根据提供的知识库，我无法准确回答此问题”，不要胡编乱造。

【检索到的参考知识】：
{context}

【用户问题】：
{query}
"""
    messages = [
        SystemMessage(content="你是一个精准、严谨的知识库问答助手。请始终用中文回答。"),
        HumanMessage(content=prompt)
    ]
    print("🧠 [大模型] 正在呼叫 DeepSeek 阅读检索结果并生成最终回答...")
    return llm.invoke(messages).content


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("====== 正在初始化 企业级 Deep RAG 智能系统 ======")
    
    embedder = DenseEmbedder(cfg['rag']['dense_model'])
    retriever = DualRetriever(embedder)
    reranker = DeepReranker(cfg['rag']['rerank_model'])
    
    corpus = [
        "PPO (近端策略优化) 是一种非常稳定的强化学习算法，适合用于智能体的训练。",
        "FAISS 是由 Facebook 开发的，专门用于高效的密集向量相似度搜索。",
        "BM25 是一种传统的稀疏检索算法，它主要基于词频(TF)和逆文档频率(IDF)的统计。",
        "RRF (倒数秩融合) 算法可以有效地把不同搜索引擎（比如向量搜索和关键词搜索）的结果混合在一起。",
        "Cross-Encoders（交叉编码器）通过自注意力机制同时处理用户问题和文档，打分极其精准，但速度较慢。"
    ]
    retriever.build_index(corpus)
    
    scheduler = PPOScheduler(cfg['scheduler']['num_nodes'], cfg['scheduler']['ppo_timesteps'])
    tracer = setup_telemetry()
    workflow = DAGWorkflow(tracer, scheduler, retriever, reciprocal_rank_fusion, reranker)

    print("\n====== 系统就绪，开始执行智能问答 ======")
    DEEPSEEK_API_KEY = "sk-c45174ee28f847f8bf59669c79cc7383"
    
    queries = [
        "如果我想做高效的向量搜索，有什么工具推荐吗？是哪个公司开发的？"
    ]

    for q in queries:
        print(f"\n======================================")
        print(f"🙋‍♂️ 收到用户原始提问: '{q}'")
        print(f"======================================")
        
        start_time = time.time()
        
        # 【关键改动】：在送入 DAG 工作流之前，先重写 Query
        with tracer.start_as_current_span("Query_Rewrite"):
            optimized_query = rewrite_query(q, DEEPSEEK_API_KEY)
        
        # 使用优化后的超级检索词去本地知识库“找答案”
        results = workflow.execute(optimized_query, corpus, cfg['rag']['top_k'], cfg['rag']['rrf_k'])
        
        print("\n🎯 [内部检索结果] 找到了以下最相关的本地知识：")
        for rank, res in enumerate(results):
            print(f"  [{rank+1}] 相关度得分: {res['score']:.4f} | 内容: {res['text']}")
            
        with tracer.start_as_current_span("LLM_Generation"):
            # 注意：生成回答时，依然传入用户原始问题 (q)，以保证回答语气自然
            final_answer = generate_answer(q, results, DEEPSEEK_API_KEY)
            
        latency = (time.time() - start_time)
        
        print(f"\n🤖 DeepSeek 最终回复 (耗时 {latency:.2f} 秒):")
        print(f"\n{final_answer}\n")

if __name__ == "__main__":
    main()