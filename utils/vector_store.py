# -*- coding: utf-8 -*-
"""向量存储模块 - 使用TF-IDF（轻量级方案，兼容Streamlit Cloud）"""
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    """向量存储管理器（TF-IDF版本）"""
    
    def __init__(self, persist_directory: str):
        """
        初始化向量存储
        
        Args:
            persist_directory: 数据持久化目录
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # 存储各课程的向量化器和文档
        self.courses: Dict[str, Dict] = {}  # {course_name: {"vectorizer": ..., "documents": ..., "metadatas": ...}}
        
        # 加载已有数据
        self._load_all_courses()
    
    def _get_course_file(self, course_name: str) -> str:
        """获取课程数据文件路径"""
        safe_name = hashlib.md5(course_name.encode()).hexdigest()[:8]
        return os.path.join(self.persist_directory, f"course_{safe_name}.pkl")
    
    def _load_all_courses(self):
        """加载所有课程数据"""
        for filename in os.listdir(self.persist_directory):
            if filename.startswith("course_") and filename.endswith(".pkl"):
                filepath = os.path.join(self.persist_directory, filename)
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        self.courses[data["course_name"]] = {
                            "vectorizer": data["vectorizer"],
                            "documents": data["documents"],
                            "metadatas": data["metadatas"],
                            "tfidf_matrix": data["tfidf_matrix"]
                        }
                except Exception as e:
                    print(f"加载课程数据失败: {e}")
    
    def _save_course(self, course_name: str):
        """保存课程数据"""
        if course_name not in self.courses:
            return
        
        data = self.courses[course_name]
        filepath = self._get_course_file(course_name)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                "course_name": course_name,
                "vectorizer": data["vectorizer"],
                "documents": data["documents"],
                "metadatas": data["metadatas"],
                "tfidf_matrix": data["tfidf_matrix"]
            }, f)
    
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
            texts = [doc["text"] for doc in documents]
            metadatas = [
                {**doc.get("metadata", {}), "file_name": file_name, "chunk_id": doc["chunk_id"]}
                for doc in documents
            ]
            
            if course_name not in self.courses:
                # 新课程，创建新的向量化器
                vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(texts)
                self.courses[course_name] = {
                    "vectorizer": vectorizer,
                    "documents": texts,
                    "metadatas": metadatas,
                    "tfidf_matrix": tfidf_matrix
                }
            else:
                # 已有课程，追加文档
                course_data = self.courses[course_name]
                all_texts = course_data["documents"] + texts
                all_metadatas = course_data["metadatas"] + metadatas
                
                # 重新训练向量化器
                vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(all_texts)
                
                self.courses[course_name] = {
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
            query: 查询文本
            top_k: 返回结果数量
        
        Returns:
            List[Dict]: 搜索结果 [{"text": "...", "score": 0.8, "metadata": {...}}, ...]
        """
        if course_name not in self.courses:
            return []
        
        course_data = self.courses[course_name]
        vectorizer = course_data["vectorizer"]
        documents = course_data["documents"]
        metadatas = course_data["metadatas"]
        tfidf_matrix = course_data["tfidf_matrix"]
        
        # 查询向量化
        query_vec = vectorizer.transform([query])
        
        # 计算相似度
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
        
        # 获取top_k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 只返回有相关性的结果
                results.append({
                    "text": documents[idx],
                    "score": float(similarities[idx]),
                    "metadata": metadatas[idx]
                })
        
        return results
    
    def delete_course(self, course_name: str) -> bool:
        """删除课程数据"""
        if course_name in self.courses:
            del self.courses[course_name]
        
        filepath = self._get_course_file(course_name)
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return True
    
    def get_course_stats(self, course_name: str) -> Dict:
        """获取课程统计信息"""
        if course_name not in self.courses:
            return {"document_count": 0}
        
        course_data = self.courses[course_name]
        return {
            "document_count": len(course_data["documents"]),
            "files": list(set(m.get("file_name", "") for m in course_data["metadatas"]))
        }
    
    def list_files(self, course_name: str) -> List[str]:
        """列出课程下的所有文件"""
        if course_name not in self.courses:
            return []
        
        files = set()
        for metadata in self.courses[course_name]["metadatas"]:
            if "file_name" in metadata:
                files.add(metadata["file_name"])
        
        return list(files)
    
    def delete_file(self, course_name: str, file_name: str) -> bool:
        """删除指定文件的所有文档"""
        if course_name not in self.courses:
            return False
        
        course_data = self.courses[course_name]
        
        # 过滤掉要删除的文件
        new_documents = []
        new_metadatas = []
        for doc, meta in zip(course_data["documents"], course_data["metadatas"]):
            if meta.get("file_name") != file_name:
                new_documents.append(doc)
                new_metadatas.append(meta)
        
        if len(new_documents) == 0:
            # 如果没有文档了，删除整个课程
            self.delete_course(course_name)
        else:
            # 重新训练向量化器
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(new_documents)
            
            self.courses[course_name] = {
                "vectorizer": vectorizer,
                "documents": new_documents,
                "metadatas": new_metadatas,
                "tfidf_matrix": tfidf_matrix
            }
            self._save_course(course_name)
        
        return True


def get_vector_store(persist_directory: str) -> VectorStore:
    """获取向量存储实例"""
    return VectorStore(persist_directory)
