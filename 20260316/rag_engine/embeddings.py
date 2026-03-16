from sentence_transformers import SentenceTransformer

#向量化封装
class DenseEmbedder:
    def __init__(self,model_name):
        self.model = SentenceTransformer(model_name)
        self.dim=self.model.get_sentence_embedding_dimension()


    def encode(self,texts):
        return self.model.encode(texts,normalize_embeddings=True)