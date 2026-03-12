import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class KnowledgeBaseManager:
    def __init__(self,persist_dir:str="./data/knowledge_base"):
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self._initialize_kb()

    def _initialize_kb(self):
        """初始化或加载向量数据库"""
        if  os.path.exists(os.path.join(self.persist_dir,"index.faiss")):
          self.vector_store=FAISS.load_local(
            self.persist_dir,
            self.embeddings,
            allow_dangerous_deserialization=True# 信任本地文件
            )
        else:
            # 初始化一些模拟的客服知识
            docs = [
                Document(page_content="退货政策：购买后7天内可无理由退换货，商品需保持原样。"),
                Document(page_content="会员等级说明：消费满1000元升级为黄金会员，享受9折优惠。"),
                Document(page_content="发货时间：一般在下单后24小时内发货，节假日顺延。")
            ]
            self.vector_store=FAISS.from_documents(docs,self.embeddings)
            self.vector_store.save_local(self.persist_dir)

            def get_retriever(self):
                """返回优化后的检索器"""
            # 性能优化：限制返回的文档数量，提高LLM处理速度并节省Token
            return self.vector_store.as_retriever(
                search_kwargs={"k": 2}  # 每次仅返回2条相关文档
            )