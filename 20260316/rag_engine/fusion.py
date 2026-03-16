# RRF 算法融合
def reciprocal_rank_fusion(dense_idx, sparse_idx, k=60, top_k=5):
    """
    RRF 算法核心：打破 FAISS (内积) 和 BM25 (TF-IDF变体) 的分数尺度壁垒。
    公式: score = 1 / (k + rank)
    """
    rrf_scores = {}

    for rank, doc_id in enumerate(dense_idx):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, doc_id in enumerate(sparse_idx):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    # 按融合分数排序，返回 top_n 个文档索引
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in fused[:top_k]]
