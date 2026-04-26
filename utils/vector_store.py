# -*- coding: utf-8 -*-
"""向量存储模块 - 使用ChromaDB"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib
import json

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class VectorStore:
    """向量存储管理器"""
    
    def __init__(self, persist_directory: str):
        """
        初始化向量存储
        
        Args:
            persist_directory: 向量数据库持久化目录
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("请先安装 chromadb: pip install chromadb")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("请先安装 sentence-transformers: pip install sentence-transformers")
        
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # 初始化embedding模型
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def _get_collection_name(self, course_name: str) -> str:
        """获取课程对应的collection名称"""
        # 使用hash确保名称合法
        return f"course_{hashlib.md5(course_name.encode()).hexdigest()[:8]}"
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本的向量嵌入"""
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def add_documents(
        self,
        course_name: str,
        documents: List[Dict],
        file_name: str
    ) -> bool:
        """
        添加文档到向量库
        
        Args:
            course_name: 课程名称
            documents: 文档片段列表 [{"text": "...", "chunk_id": 0, "metadata": {...}}, ...]
            file_name: 原始文件名
        
        Returns:
            bool: 是否成功
        """
        try:
            collection_name = self._get_collection_name(course_name)
            
            # 获取或创建collection
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"course": course_name, "file": file_name}
            )
            
            # 准备数据
            texts = [doc["text"] for doc in documents]
            ids = [f"{file_name}_{doc['chunk_id']}" for doc in documents]
            metadatas = [
                {**doc.get("metadata", {}), "file_name": file_name, "chunk_id": doc["chunk_id"]}
                for doc in documents
            ]
            
            # 计算embeddings
            embeddings = self._get_embeddings(texts)
            
            # 添加到collection
            collection.add(
                documents=texts,
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            return True
            
        except Exception as e:
            print(f"添加文档失败: {e}")
            return False
    
    def search(
        self,
        course_name: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        搜索相关文档
        
        Args:
            course_name: 课程名称
            query: 搜索查询
            top_k: 返回数量
        
        Returns:
            List[Dict]: 相关文档列表
        """
        try:
            collection_name = self._get_collection_name(course_name)
            collection = self.client.get_collection(name=collection_name)
            
            # 计算查询向量
            query_embedding = self._get_embeddings([query])[0]
            
            # 搜索
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # 整理结果
            documents = []
            if results["documents"] and len(results["documents"]) > 0:
                for i, doc in enumerate(results["documents"][0]):
                    documents.append({
                        "text": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0
                    })
            
            return documents
            
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def delete_file(self, course_name: str, file_name: str) -> bool:
        """删除指定文件的所有向量"""
        try:
            collection_name = self._get_collection_name(course_name)
            collection = self.client.get_collection(name=collection_name)
            
            # 获取该文件的所有ID
            all_data = collection.get()
            ids_to_delete = [
                id_ for id_ in all_data["ids"]
                if id_.startswith(f"{file_name}_")
            ]
            
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
            
            return True
            
        except Exception as e:
            print(f"删除向量失败: {e}")
            return False
    
    def get_collection_info(self, course_name: str) -> Dict:
        """获取collection信息"""
        try:
            collection_name = self._get_collection_name(course_name)
            collection = self.client.get_collection(name=collection_name)
            return {
                "name": collection_name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception:
            return {}
    
    def list_all_files(self, course_name: str) -> List[str]:
        """列出课程中的所有文件"""
        try:
            collection_name = self._get_collection_name(course_name)
            collection = self.client.get_collection(name=collection_name)
            all_data = collection.get()
            
            files = set()
            for id_ in all_data["ids"]:
                # ID格式: filename_chunkid
                file_name = "_".join(id_.split("_")[:-1])
                files.add(file_name)
            
            return list(files)
        except Exception:
            return []


def get_vector_store(course_name: str, base_dir: str) -> VectorStore:
    """获取课程对应的向量存储"""
    persist_directory = os.path.join(base_dir, "vector_db", course_name)
    return VectorStore(persist_directory)
