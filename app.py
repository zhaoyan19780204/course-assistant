# -*- coding: utf-8 -*-
"""课程伴学助教 - Streamlit单文件版本"""
import streamlit as st
import os
import json
import pickle
import hashlib
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ==================== 配置模块 ====================
CONFIG_FILE = Path("data/config.json")
COURSES_DIR = Path("data/courses")

MODEL_CONFIG = {
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat"
    },
    "qwen": {
        "name": "通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-turbo", "qwen-plus", "qwen-max"],
        "default_model": "qwen-turbo"
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "default_model": "gpt-4o-mini"
    }
}

def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_config(config: dict) -> bool:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return True

def get_course_path(course_name: str) -> Path:
    path = COURSES_DIR / course_name
    path.mkdir(parents=True, exist_ok=True)
    return path

def list_courses() -> List[str]:
    if not COURSES_DIR.exists():
        return []
    return [d.name for d in COURSES_DIR.iterdir() if d.is_dir()]

def delete_course(course_name: str) -> bool:
    import shutil
    path = COURSES_DIR / course_name
    if path.exists():
        shutil.rmtree(path)
    return True


# ==================== 文档解析模块 ====================
def parse_docx(file_path: str) -> str:
    from docx import Document
    doc = Document(file_path)
    texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(texts)

def parse_txt(file_path: str) -> str:
    for encoding in ['utf-8', 'gbk', 'gb2312']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except:
            continue
    return ""

def parse_pdf(file_path: str) -> str:
    import pdfplumber
    texts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return "\n".join(texts)

def parse_document(file_path: str) -> Tuple[str, str]:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".docx":
        return parse_docx(file_path), "word"
    elif suffix == ".txt":
        return parse_txt(file_path), "txt"
    elif suffix == ".pdf":
        return parse_pdf(file_path), "pdf"
    else:
        raise Exception(f"不支持的格式: {suffix}")

def chunk_text(text: str, chunk_size: int = 500) -> List[Dict]:
    """简单切分文本"""
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = []
    current_size = 0
    current_section = ""
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        
        # 检测章节标题
        import re
        if re.match(r'^[一二三四五六七八九十]+、', p) or re.match(r'^第.+[章节]', p):
            current_section = p
        
        if current_size + len(p) > chunk_size and current_chunk:
            chunks.append({
                "text": "\n".join(current_chunk),
                "chunk_id": len(chunks),
                "metadata": {"section": current_section}
            })
            current_chunk = []
            current_size = 0
        
        current_chunk.append(p)
        current_size += len(p)
    
    if current_chunk:
        chunks.append({
            "text": "\n".join(current_chunk),
            "chunk_id": len(chunks),
            "metadata": {"section": current_section}
        })
    
    return chunks

def get_file_type_label(suffix: str) -> str:
    labels = {".docx": "Word", ".txt": "文本", ".pdf": "PDF"}
    return labels.get(suffix.lower(), "未知")


# ==================== 向量存储模块 ====================
class SimpleVectorStore:
    """简单的向量存储 - 使用session_state持久化"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用 session_state 存储数据
        if "vector_store_data" not in st.session_state:
            st.session_state.vector_store_data = {}
        
        self.courses = st.session_state.vector_store_data
    
    def _get_file(self, course_name: str) -> Path:
        safe = hashlib.md5(course_name.encode()).hexdigest()[:8]
        return self.data_dir / f"course_{safe}.pkl"
    
    def add_documents(self, course_name: str, documents: List[Dict], file_name: str) -> bool:
        try:
            texts = [d["text"] for d in documents]
            metas = [{**d.get("metadata", {}), "file_name": file_name} for d in documents]
            
            if course_name not in self.courses:
                self.courses[course_name] = {"documents": texts, "metadatas": metas}
            else:
                self.courses[course_name]["documents"].extend(texts)
                self.courses[course_name]["metadatas"].extend(metas)
            
            # 更新 session_state
            st.session_state.vector_store_data = self.courses
            
            # 也保存到文件（备份）
            try:
                data = {
                    "course_name": course_name,
                    "documents": self.courses[course_name]["documents"],
                    "metadatas": self.courses[course_name]["metadatas"]
                }
                with open(self._get_file(course_name), 'wb') as f:
                    pickle.dump(data, f)
            except:
                pass
            
            return True
        except Exception as e:
            st.error(f"添加文档失败: {e}")
            return False
    
    def search(self, course_name: str, query: str, top_k: int = 5) -> List[Dict]:
        if course_name not in self.courses:
            return []
        
        data = self.courses[course_name]
        docs = data["documents"]
        metas = data["metadatas"]
        
        if not docs:
            return []
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer(max_features=3000)
            all_texts = docs + [query]
            matrix = vectorizer.fit_transform(all_texts)
            sims = cosine_similarity(matrix[-1:], matrix[:-1])[0]
            top_idx = np.argsort(sims)[::-1][:top_k]
            
            results = []
            for i in top_idx:
                if sims[i] > 0:
                    results.append({
                        "text": docs[i],
                        "score": float(sims[i]),
                        "metadata": metas[i] if i < len(metas) else {}
                    })
            return results
        except Exception as e:
            st.error(f"检索失败: {e}")
            return []
    
    def get_stats(self, course_name: str) -> Dict:
        if course_name not in self.courses:
            return {"count": 0}
        return {"count": len(self.courses[course_name]["documents"])}
    
    def list_files(self, course_name: str) -> List[str]:
        if course_name not in self.courses:
            return []
        return list(set(m.get("file_name", "") for m in self.courses[course_name]["metadatas"]))
    
    def delete_file(self, course_name: str, file_name: str) -> bool:
        if course_name not in self.courses:
            return False
        
        data = self.courses[course_name]
        new_docs = []
        new_metas = []
        
        for doc, meta in zip(data["documents"], data["metadatas"]):
            if meta.get("file_name") != file_name:
                new_docs.append(doc)
                new_metas.append(meta)
        
        data["documents"] = new_docs
        data["metadatas"] = new_metas
        self._save(course_name)
        return True


# ==================== LLM客户端 ====================
def call_llm(provider: str, api_key: str, model: str, messages: List[Dict]) -> str:
    """调用大模型API"""
    import openai
    
    config = MODEL_CONFIG.get(provider, MODEL_CONFIG["deepseek"])
    client = openai.OpenAI(api_key=api_key, base_url=config["base_url"])
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=2000
    )
    
    return response.choices[0].message.content

def build_prompt(question: str, contexts: List[Dict], course_name: str) -> List[Dict]:
    """构建RAG提示词"""
    context_text = "\n\n".join([
        f"【参考资料{i+1}】{c['metadata'].get('section', '')}\n{c['text']}"
        for i, c in enumerate(contexts)
    ])
    
    system_prompt = f"""你是课程「{course_name}」的学习助手。
请基于以下参考资料回答问题，并在回答末尾标注内容来源。

参考资料：
{context_text}

要求：
1. 回答要准确、清晰
2. 如果参考资料中没有相关信息，请说明
3. 在回答末尾用以下格式标注来源：
   📖 来源：课件名 - 章节名
"""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]


# ==================== Streamlit应用 ====================
st.set_page_config(
    page_title="课程伴学助教",
    page_icon="📚",
    layout="wide"
)

# 初始化
if "current_course" not in st.session_state:
    st.session_state.current_course = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "config" not in st.session_state:
    st.session_state.config = load_config()

# 侧边栏
with st.sidebar:
    st.markdown("### ⚙️ 模型配置")
    
    provider = st.selectbox(
        "选择大模型",
        options=list(MODEL_CONFIG.keys()),
        format_func=lambda x: MODEL_CONFIG[x]["name"],
        index=0
    )
    
    api_key = st.text_input("API Key", type="password", value=st.session_state.config.get("api_key", ""))
    
    model = st.selectbox(
        "选择模型",
        options=MODEL_CONFIG[provider]["models"],
        index=0
    )
    
    if st.button("保存配置", type="primary"):
        st.session_state.config = {"provider": provider, "api_key": api_key, "model": model}
        save_config(st.session_state.config)
        st.success("✅ 配置已保存")
    
    st.divider()
    st.markdown("### 📚 课程管理")
    
    new_course = st.text_input("新课程名称")
    if st.button("创建课程") and new_course:
        get_course_path(new_course)
        st.session_state.current_course = new_course
        st.success(f"✅ 课程「{new_course}」已创建")
        st.rerun()
    
    courses = list_courses()
    for course in courses:
        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button(f"📖 {course}", key=f"sel_{course}"):
                st.session_state.current_course = course
                st.session_state.messages = []
                st.rerun()
        with c2:
            if st.button("🗑️", key=f"del_{course}"):
                delete_course(course)
                if st.session_state.current_course == course:
                    st.session_state.current_course = None
                st.rerun()

# 主界面
if not st.session_state.current_course:
    st.info("👈 请先在侧边栏选择或创建课程")
    st.stop()

course_name = st.session_state.current_course
course_dir = get_course_path(course_name)

tab1, tab2 = st.tabs(["📄 课件管理", "💬 问答助手"])

with tab1:
    st.markdown(f"### 当前课程：**{course_name}**")
    
    # 上传课件
    uploaded_files = st.file_uploader(
        "上传课件（支持 Word/TXT/PDF）",
        type=["docx", "txt", "pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("上传并处理"):
        vs = SimpleVectorStore(str(COURSES_DIR))
        progress = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            # 保存文件
            file_path = course_dir / file.name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            try:
                # 解析文档
                text, ftype = parse_document(str(file_path))
                chunks = chunk_text(text)
                vs.add_documents(course_name, chunks, file.name)
                
                # 保存元数据
                meta = {"name": file.name, "type": ftype, "chunks": len(chunks)}
                with open(course_dir / f"{file.name}.meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False)
                
            except Exception as e:
                st.error(f"处理 {file.name} 失败: {e}")
            
            progress.progress((i + 1) / len(uploaded_files))
        
        st.success(f"✅ 已处理 {len(uploaded_files)} 个文件")
        st.rerun()
    
    # 课件列表
    st.markdown("#### 已上传课件")
    vs = SimpleVectorStore(str(COURSES_DIR))
    stats = vs.get_stats(course_name)
    st.metric("文档片段数", stats["count"])
    
    for f in vs.list_files(course_name):
        st.markdown(f"📄 {f}")

with tab2:
    st.markdown("### 💬 课程问答")
    
    if not st.session_state.config.get("api_key"):
        st.warning("⚠️ 请先在侧边栏配置 API Key")
        st.stop()
    
    # 聊天记录
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 输入框
    if question := st.chat_input("输入你的问题..."):
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                # 检索相关内容
                vs = SimpleVectorStore(str(COURSES_DIR))
                contexts = vs.search(course_name, question, top_k=3)
                
                if contexts:
                    # 构建提示词
                    messages = build_prompt(question, contexts, course_name)
                    
                    # 调用大模型
                    try:
                        answer = call_llm(
                            st.session_state.config["provider"],
                            st.session_state.config["api_key"],
                            st.session_state.config["model"],
                            messages
                        )
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"AI回答失败: {e}")
                        answer = f"回答失败: {e}"
                else:
                    answer = "抱歉，没有找到相关的课件内容。请先上传相关课件。"
                    st.warning(answer)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
    
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()
