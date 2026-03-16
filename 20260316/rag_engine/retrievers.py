import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import jieba  # 引入中文分词神器

class DualRetriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.faiss_index = faiss.IndexFlatIP(self.embedder.dim)#使用内积相似度
        self.bm25 = None
        self.corpus = []

    def build_index(self, corpus):
        self.corpus = corpus
        # 【修改点1】：使用 jieba 把中文句子切分成词语
        tokenized_corpus = [jieba.lcut(doc) for doc in corpus]#切分后的语料列表，供 BM25 使用
        self.bm25 = BM25Okapi(tokenized_corpus)#构建 BM25 索引
        
        embs = self.embedder.encode(corpus)#构建向量索引
        self.faiss_index.add(np.array(embs).astype('float32'))#把向量添加到 FAISS 索引中

    def search_dense(self, query, top_n):
        q_emb = self.embedder.encode([query])#把用户的中文提问也向量化
        scores, indices = self.faiss_index.search(np.array(q_emb).astype('float32'), top_n)#在 FAISS 索引中搜索最相似的 top_n 个文档
        return indices[0], scores[0]#返回最相似的文档索引和对应的相似度分数

    def search_sparse(self, query, top_n):
        # 【修改点2】：对用户的中文提问也使用 jieba 切分
        tokenized_query = jieba.lcut(query)
        scores = self.bm25.get_scores(tokenized_query)#使用 BM25 计算每个文档与用户提问的相关性分数
        indices = np.argsort(scores)[::-1][:top_n]#返回相关性最高的 top_n 个文档索引
        return indices, scores[indices]#返回最相关的文档索引和对应的 BM25 分数