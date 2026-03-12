import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class KnowledgeBaseManager:
    def __init__(self, data_path: str = "data/knowledge_base/faq.json", persist_dir: str = "data/knowledge_base/faiss_index"):
        # 获取当前运行的根目录，确保路径绝对正确
        self.base_dir = os.getcwd() 
        self.data_path = os.path.join(self.base_dir, data_path)
        self.persist_dir = os.path.join(self.base_dir, persist_dir)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        self._initialize_kb()

    def _initialize_kb(self):
        """初始化或加载向量数据库"""
        index_path = os.path.join(self.persist_dir, "index.faiss")
        
        if os.path.exists(index_path):
            print("📦 正在加载已存在的本地向量库...")
            self.vector_store = FAISS.load_local(
                self.persist_dir, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print(f"🔨 准备从 {self.data_path} 读取数据...")
            docs = []
            
            # 1. 强力校验：文件到底存不存在？
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"❌ 找不到知识库文件！程序去这里找了：{self.data_path}")
                
            # 2. 读取数据
            with open(self.data_path, "r", encoding="utf-8") as f:
                faqs = json.load(f)
                for faq in faqs:
                    content = f"问题：{faq['question']}\n答案：{faq['answer']}"
                    docs.append(Document(page_content=content, metadata={"source": "faq.json"}))
            
            # 3. 强力校验：数据是不是空的？
            if not docs:
                raise ValueError("❌ JSON 文件是空的，没有读取到任何内容！")
                
            print(f"✅ 成功读取到 {len(docs)} 条知识，正在灌入 FAISS 向量库 (这可能需要几秒钟)...")
            
            # 4. 灌入数据库
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.save_local(self.persist_dir)
            print("✅ 向量库构建完成！")

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": 3})