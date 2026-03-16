import networkx as nx

class DAGWorkflow:
    def __init__(self, tracer,scheduler,retriever_sys,fusion_fn,reranker_sys):
        self.tracer = tracer
        self.scheduler = scheduler
        self.retriever = retriever_sys
        self.fusion = fusion_fn
        self.reranker = reranker_sys
        

        #构建有向无环图（DAG）
        self.dag=nx.DiGraph()
        self.dag.add_edges_from(
            [
                ("Schedule", "Recall"),
                ("Recall", "Fusion"),
                ("Fusion", "Rerank")
            ]
        )
    

    def execute(self,query,corpus,top_k,rrf_k):
        with self.tracer.start_as_current_span("Total_Workflow"):

            with self.tracer.start_as_current_span("Node_Schedule") as span:
                node = self.scheduler.get_best_node()
                span.set_attribute("target_node", node)
                print(f"✅ [调度] 任务分配至 Node-{node}")

            with self.tracer.start_as_current_span("Dual_Recall"):
                # 扩大召回基数保证召回率
                dense_idx, _ = self.retriever.search_dense(query, top_n=top_k*2)
                sparse_idx, _ = self.retriever.search_sparse(query, top_n=top_k*2)
                print(f"✅ [召回] FAISS 与 BM25 多路召回完成")

            with self.tracer.start_as_current_span("RRF_Fusion"):
                fused_idx = self.fusion(dense_idx, sparse_idx, k=rrf_k, top_k=top_k*2)
                print(f"✅ [融合] RRF 算法计算完成")

            with self.tracer.start_as_current_span("Cross_Encoder_Rerank"):
                final_results = self.reranker.rerank(query, fused_idx, corpus)
                print(f"✅ [精排] Cross-Encoder 深度打分完成")
                
        return final_results[:top_k]