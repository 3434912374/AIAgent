# Cross-Encoder 精排

from sentence_transformers import CrossEncoder


class DeepReranker:
    def __init__(self, model_name):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates,corpus):
        pairs=[[query,corpus[idx]] for  idx in candidates]#构建查询-候选对
        socres=self.model.predict(pairs)  #交叉编码器打分

        results=[{"doc_id":candidates[i],
                  "text":corpus[candidates[i]],
                  "score":float(socres[i])} for i in range(len(candidates))]  #构建结果列表
        return sorted(results,key=lambda x:x["score"],reverse=True)  #按分数排序返回结果