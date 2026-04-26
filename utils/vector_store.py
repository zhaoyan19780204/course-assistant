# -*- coding: utf-8 -*-
"""向量存储模块 - 使用TF-IDF（轻量级方案，兼容Streamlit Cloud）"""
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib
import numpy as np

# 延迟导入sklearn，避免启动时加载
_vectorizer = None
_cosine_similarity = None

def _get_sklearn():
    """延迟加载sklearn"""
    global _vectorizer, _cosine_similarity
    if _vectorizer is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        _vectorizer = TfidfVectorizer
        _cosine_similarity = cosine_similarity
    return _vectorizer, _cosine_similarity


class VectorStore:
    """向量存储管理器（TF-IDF版本）"""
    
    def __init__(self, persist_directory: str):
        """初始化"""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.courses: Dict[str, Dict] = {}
        self._load_all_courses()
    
    def _get_course_file(self, course_name: str) -> str:
        safe_name = hashlib.md5(course_name.encode()).hexdigest()[:8]
        return os.path.join(self.persist_directory, f"course_{safe_name}.pkl")
    
    def _load_all_courses(self):
        """加载所有课程数据"""
        if not os.path.exists(self.persist_directory):
            return
        for filename in os.listdir(self.persist_directory):
            if filename.startswith("course_") and filename.endswith(".pkl"):
                filepath = os.path.join(self.persist_directory, filename)
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        self.courses[data["course_name"]] = data
                except Exception as e:
                    print(f"加载课程数据失败: {e}")
    
    def _save_course(self, course_name: str):
        """保存课程数据"""
        if course_name not in self.courses:
            return
        filepath = self._get_course_file(course_name)
        with open(filepath, 'wb') as f:
            pickle.dump(self.courses[course_name], f)
    
    def add_documents(self, course_name: str, documents: List[Dict], file_name: str) -> bool:
        """添加文档"""
        try:
            TfidfVectorizer, cosine_similarity = _get_sklearn()
            
            texts = [doc["text"] for doc in documents]
            metadatas = [
                {**doc.get("metadata", {}), "file_name": file_name, "chunk_id": doc["chunk_id"]}
                for doc in documents
            ]
            
            if course_name not in self.courses:
                vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(texts)
                self.courses[course_name] = {
                    "course_name": course_name,
                    "vectorizer": vectorizer,
                    "documents": texts,
                    "metadatas": metadatas,
                    "tfidf_matrix": tfidf_matrix
                }
            else:
                course_data = self.courses[course_name]
                all_texts = course_data["documents"] + texts
                all_metadatas = course_data["metadatas"] + metadatas
                vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(all_texts)
                self.courses[course_name] = {
                    "course_name": course_name,
                    "vectorizer": vectorizer,
                    "documents": all_texts,
                    "metadatas": all_metadatas,
                    "tfidf_matrix": tfidf_matrix
                }
            
            self._save_course(course_name)
            return True
        except Exception as e:
            print(f"添加文档失败: {e}")
            return False
    
    def search(self, course_name: str, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相关文档"""
        if course_name not in self.courses:
            return []
        
        try:
            TfidfVectorizer, cosine_similarity = _get_sklearn()
            course_data = self.courses[course_name]
            vectorizer = course_data["vectorizer"]
            documents = course_data["documents"]
            metadatas = course_data["metadatas"]
            tfidf_matrix = course_data["tfidf_matrix"]
            
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    results.append({
                        "text": documents[idx],
                        "score": float(similarities[idx]),
                        "metadata": metadatas[idx]
                    })
            return results
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def delete_course(self, course_name: str) -> bool:
        """删除课程"""
        if course_name in self.courses:
            del self.courses[course_name]
        filepath = self._get_course_file(course_name)
        if os.path.exists(filepath):
            os.remove(filepath)
        return True
    
    def get_course_stats(self, course_name: str) -> Dict:
        """获取课程统计"""
        if course_name not in self.courses:
            return {"document_count": 0}
        return {"document_count": len(self.courses[course_name]["documents"])}
    
    def get_collection_info(self, course_name: str) -> Optional[Dict]:
        """获取collection信息（兼容app.py调用）"""
        if course_name not in self.courses:
            return None
        course_data = self.courses[course_name]
        return {"count": len(course_data["documents"])}
    
    def list_files(self, course_name: str) -> List[str]:
        """列出课程下的所有文件"""
        if course_name not in self.courses:
            return []
        files = set()
        for metadata in self.courses[course_name]["metadatas"]:
            if "file_name" in metadata:
                files.add(metadata["file_name"])
        return list(files)
    
    def list_all_files(self, course_name: str) -> List[str]:
        """列出所有文件（兼容app.py调用）"""
        return self.list_files(course_name)
    
    def delete_file(self, course_name: str, file_name: str) -> bool:
        """删除指定文件"""
        if course_name not in self.courses:
            return False
        
        course_data = self.courses[course_name]
        new_documents = []
        new_metadatas = []
        
        for doc, meta in zip(course_data["documents"], course_data["metadatas"]):
            if meta.get("file_name") != file_name:
                new_documents.append(doc)
                new_metadatas.append(meta)
        
        if len(new_documents) == 0:
            self.delete_course(course_name)
        else:
            TfidfVectorizer, _ = _get_sklearn()
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(new_documents)
            self.courses[course_name] = {
                "course_name": course_name,
                "vectorizer": vectorizer,
                "documents": new_documents,
                "metadatas": new_metadatas,
                "tfidf_matrix": tfidf_matrix
            }
            self._save_course(course_name)
        return True


# 全局实例缓存
_vector_store_instances: Dict[str, VectorStore] = {}

def get_vector_store(course_name: str = "", persist_directory: str = "") -> VectorStore:
    """获取向量存储实例"""
    # 兼容两种调用方式：
    # 1. get_vector_store(course_name, persist_dir)
    # 2. get_vector_store(persist_dir)
    if not persist_directory:
        persist_directory = course_name
        course_name = ""
    
    if persist_directory not in _vector_store_instances:
        _vector_store_instances[persist_directory] = VectorStore(persist_directory)
    
    return _vector_store_instances[persist_directory]
