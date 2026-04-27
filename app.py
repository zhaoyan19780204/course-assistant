# -*- coding: utf-8 -*-
"""小书童 - 文件持久化版本 + 中文分词优化"""
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
DATA_DIR = Path("data")
CONFIG_FILE = DATA_DIR / "config.json"
COURSES_DIR = DATA_DIR / "courses"
VECTORS_DIR = DATA_DIR / "vectors"

DATA_DIR.mkdir(parents=True, exist_ok=True)
COURSES_DIR.mkdir(parents=True, exist_ok=True)
VECTORS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIG = {
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-reasoner", "deepseek-chat"],
        "default_model": "deepseek-reasoner"
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
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return True

def get_course_path(course_name: str) -> Path:
    path = COURSES_DIR / hashlib.md5(course_name.encode()).hexdigest()[:8]
    path.mkdir(parents=True, exist_ok=True)
    name_file = path / ".name"
    if not name_file.exists():
        with open(name_file, "w", encoding="utf-8") as f:
            f.write(course_name)
    return path

def list_courses() -> List[str]:
    if not COURSES_DIR.exists():
        return []
    courses = []
    for d in COURSES_DIR.iterdir():
        if d.is_dir():
            name_file = d / ".name"
            if name_file.exists():
                with open(name_file, "r", encoding="utf-8") as f:
                    courses.append(f.read().strip())
    return courses

def delete_course(course_name: str) -> bool:
    import shutil
    safe = hashlib.md5(course_name.encode()).hexdigest()[:8]
    path = COURSES_DIR / safe
    vec_path = VECTORS_DIR / f"{safe}.pkl"
    if path.exists():
        shutil.rmtree(path)
    if vec_path.exists():
        vec_path.unlink()
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
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = []
    current_size = 0
    current_section = ""
    
    import re
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
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


# ==================== 向量存储模块 ====================
class FileVectorStore:
    def __init__(self, course_name: str):
        self.course_name = course_name
        self.safe_name = hashlib.md5(course_name.encode()).hexdigest()[:8]
        self.data_file = VECTORS_DIR / f"{self.safe_name}.pkl"
        self.data = self._load()
    
    def _load(self) -> Dict:
        if self.data_file.exists():
            with open(self.data_file, 'rb') as f:
                return pickle.load(f)
        return {"documents": [], "metadatas": []}
    
    def _save(self):
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.data, f)
    
    def add_documents(self, documents: List[Dict], file_name: str) -> bool:
        try:
            for d in documents:
                self.data["documents"].append(d["text"])
                self.data["metadatas"].append({
                    **d.get("metadata", {}),
                    "file_name": file_name
                })
            self._save()
            return True
        except Exception as e:
            st.error(f"添加文档失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        docs = self.data.get("documents", [])
        metas = self.data.get("metadatas", [])
        if not docs:
            return []
        try:
            import jieba
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            def tokenize(text):
                return ' '.join(jieba.cut(text))
            
            tokenized_docs = [tokenize(d) for d in docs]
            tokenized_query = tokenize(query)
            
            vectorizer = TfidfVectorizer(max_features=3000)
            all_texts = tokenized_docs + [tokenized_query]
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
    
    def get_stats(self) -> Dict:
        return {"count": len(self.data.get("documents", []))}
    
    def list_files(self) -> List[str]:
        return list(set(m.get("file_name", "") for m in self.data.get("metadatas", [])))


# ==================== LLM客户端 ====================
def call_llm(provider: str, api_key: str, model: str, messages: List[Dict]) -> str:
    import openai
    config = MODEL_CONFIG.get(provider, MODEL_CONFIG["deepseek"])
    client = openai.OpenAI(api_key=api_key.strip(), base_url=config["base_url"])
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=2000
    )
    return response.choices[0].message.content

def build_prompt(question: str, contexts: List[Dict], course_name: str) -> List[Dict]:
    context_text = "\n\n".join([
        f"【参考资料{i+1}】{c['metadata'].get('section', '')}\n{c['text']}"
        for i, c in enumerate(contexts)
    ])
    system_prompt = f"""你是课程「{course_name}」的小书童学习助手。
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
    page_title="小书童",
    page_icon="📚",
    layout="wide"
)

# 深色主题样式
st.markdown("""
<style>


/* 强制深色背景 - 覆盖所有主内容区 */
section[data-testid="stMain"],
section[data-testid="stMain"] > div,
section[data-testid="stMain"] > div > div,
.stMainBlockContainer,
.main,
.main > div,
[data-testid="stMainBlockContainer"],
[data-testid="stMainBlockContainer"] > div {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
}

/* 标签页内容区 */
.stTabs [data-testid="stVerticalBlock"],
.stTabs > div > div > div {
    background: transparent !important;
}

/* 问答区域背景 */
div[data-testid="stVerticalBlock"] {
    background: transparent !important;
}

/* 整体深色背景 */
.main .block-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem;
    min-height: 100vh;
}

/* 侧边栏样式 */
section[data-testid='stSidebar'] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}

section[data-testid='stSidebar'] .element-container {
    color: #e8e8e8;
}

/* 输入框样式 */
div[data-testid='stChatInput'] textarea {
    min-height: 120px !important;
    font-size: 16px !important;
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e8e8e8 !important;
    border-radius: 12px !important;
}

div[data-testid='stChatInput'] textarea:focus {
    border-color: #e94560 !important;
    box-shadow: 0 0 20px rgba(233,69,96,0.3) !important;
}

/* 对话消息样式 */
div[data-testid='stChatMessage'] {
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 16px;
    border: 1px solid rgba(255,255,255,0.05);
}

div[data-testid='stChatMessage'] p {
    color: #e8e8e8 !important;
}

/* 用户消息特殊样式 */
div[data-testid='stChatMessage']:has([data-testid='stChatMessageAvatarUser']) {
    background: linear-gradient(135deg, rgba(233,69,96,0.1), rgba(233,69,96,0.05));
    border-left: 3px solid #e94560;
}

/* AI消息特殊样式 */
div[data-testid='stChatMessage']:has([data-testid='stChatMessageAvatarAssistant']) {
    background: linear-gradient(135deg, rgba(15,52,96,0.3), rgba(22,33,62,0.3));
    border-left: 3px solid #0f3460;
}

/* 标题样式 */
h1, h2, h3 {
    color: #e8e8e8 !important;
    font-weight: 600 !important;
}

/* 标签页样式 */
button[data-testid='stBaseButton-secondary'] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e8e8e8 !important;
}

button[data-testid='stBaseButton-secondary']:hover {
    background: rgba(233,69,96,0.2) !important;
    border-color: #e94560 !important;
}

/* 隐藏默认的Streamlit元素 */
section[data-testid='stSidebar'] > div > div:nth-child(1) {
    display: none;
}

/* 课程按钮样式 */
button[kind="secondary"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 8px !important;
}

/* 滚动条样式 */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255,255,255,0.05);
}

::-webkit-scrollbar-thumb {
    background: rgba(233,69,96,0.3);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(233,69,96,0.5);
}

/* 提示文字颜色 */
.stMarkdown p {
    color: #b8b8b8 !important;
}

/* 信息框样式 */
div[data-testid='stInfo'] {
    background: rgba(15,52,96,0.3) !important;
    border: 1px solid rgba(233,69,96,0.3) !important;
}

div[data-testid='stSuccess'] {
    background: rgba(15,52,96,0.3) !important;
}

div[data-testid='stWarning'] {
    background: rgba(233,69,96,0.2) !important;
}

/* 主内容区背景 */
.main, .main > div, section[data-testid='stMain'] {
    background: transparent !important;
}

/* 标签页容器背景 */
.stTabs, .stTabs > div, .stTabs [data-testid='stVerticalBlock'] {
    background: transparent !important;
}

/* 激活的标签页 */
.stTabs button[aria-selected="true"] {
    background: rgba(233,69,96,0.3) !important;
    border-color: #e94560 !important;
}

/* 文件上传区域 */
section[data-testid='stFileUploader'] {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 12px;
    padding: 10px;
}

/* Metric样式 */
div[data-testid='stMetric'] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px;
    padding: 16px;
}

div[data-testid='stMetric'] label {
    color: #b8b8b8 !important;
}

div[data-testid='stMetric'] [data-testid='stMetricValue'] {
    color: #e94560 !important;
}

/* 文本输入框 */
input[type="text"], .stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e8e8e8 !important;
}

/* 选择框 */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

div[data-baseweb="select"] span {
    color: #e8e8e8 !important;
}

/* 分割线 */
hr, .stDivider {
    border-color: rgba(255,255,255,0.1) !important;
}



/* 主内容区背景 */
.main, .main > div, section[data-testid='stMain'] {
    background: transparent !important;
}

/* 标签页容器背景 */
.stTabs, .stTabs > div, .stTabs [data-testid='stVerticalBlock'] {
    background: transparent !important;
}

/* 标签页面板背景 */
div[data-testid='stVerticalBlock'] > div:has(> div[data-testid='stChatInput']) {
    background: transparent !important;
}

/* 激活的标签页 */
.stTabs button[aria-selected="true"] {
    background: rgba(233,69,96,0.3) !important;
    border-color: #e94560 !important;
}

/* 文件上传区域 */
section[data-testid='stFileUploader'] {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 12px;
    padding: 10px;
}

/* Metric样式 */
div[data-testid='stMetric'] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px;
    padding: 16px;
}

div[data-testid='stMetric'] label {
    color: #b8b8b8 !important;
}

div[data-testid='stMetric'] [data-testid='stMetricValue'] {
    color: #e94560 !important;
}

/* 文本输入框 */
input[type="text"], .stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e8e8e8 !important;
}

/* 选择框 */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

div[data-baseweb="select"] span {
    color: #e8e8e8 !important;
}

/* 分割线 */
hr, .stDivider {
    border-color: rgba(255,255,255,0.1) !important;
}

/* Expander样式 */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    color: #e8e8e8 !important;
}


/* 覆盖所有可能的白色背景区域 */
.stApp, .stApp > div, .stApp > header, .stApp > footer,
header[data-testid="stHeader"], footer[data-testid="stFooter"],
[data-testid="stHeader"], [data-testid="stFooter"],
.stDecoration, .stDeployButton,
section[data-testid="stSidebar"] + div,
.stAppViewMainViewBlockContainer,
[class*="stApp"], [class*="streamlit"] {
    background: transparent !important;
    background-color: transparent !important;
}

/* 顶部header区域 */
header {
    background: rgba(26, 26, 46, 0.95) !important;
    border-bottom: 1px solid rgba(255,255,255,0.1) !important;
}

/* 底部区域 */
footer, [data-testid="stFooter"] {
    background: rgba(26, 26, 46, 0.95) !important;
}

/* 全局背景 */
html, body, .stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
}


/* 输入框字体颜色 - 必须是浅色才能在深色背景上看见 */
div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] textarea::placeholder,
.stChatInput textarea,
.stChatInput textarea::placeholder {
    color: #e8e8e8 !important;
    -webkit-text-fill-color: #e8e8e8 !important;
}

/* placeholder颜色 */
div[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255,255,255,0.5) !important;
}


/* 输入框 - 浅色背景 + 深色字体 */
div[data-testid="stChatInput"] textarea,
.stChatInput textarea {
    background: rgba(255,255,255,0.95) !important;
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(0,0,0,0.4) !important;
}

</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "config" not in st.session_state:
    st.session_state.config = load_config()

# 默认选择第一个课程
courses = list_courses()
if "current_course" not in st.session_state:
    if courses:
        st.session_state.current_course = courses[0]
    else:
        st.session_state.current_course = None

with st.sidebar:
    # 模型配置（收起状态）
    with st.expander("⚙️ 模型配置", expanded=False):
        provider = st.selectbox(
            "选择大模型",
            options=list(MODEL_CONFIG.keys()),
            format_func=lambda x: MODEL_CONFIG[x]["name"],
            index=0
        )
        
        # 默认选择 deepseek-reasoner
        model_list = MODEL_CONFIG[provider]["models"]
        default_model = MODEL_CONFIG[provider]["default_model"]
        default_index = model_list.index(default_model) if default_model in model_list else 0
        
        api_key = st.text_input("API密钥", type="password", value=st.session_state.config.get("api_key", ""))
        model = st.selectbox(
            "选择模型",
            options=model_list,
            index=default_index
        )
        if st.button("保存配置", type="primary"):
            st.session_state.config = {"provider": provider, "api_key": api_key.strip(), "model": model}
            save_config(st.session_state.config)
            st.success("✅ 配置已保存")
    
    st.divider()
    st.markdown("### 📚 课程管理 (v0.5.4)")
    new_course = st.text_input("新课程名称")
    if st.button("创建课程") and new_course:
        get_course_path(new_course)
        st.session_state.current_course = new_course
        st.success(f"✅ 课程「{new_course}」已创建")
        st.rerun()
    
    courses = list_courses()
    for idx, course in enumerate(courses, 1):
        c1, c2 = st.columns([3, 1])
        with c1:
            # 课程前加序号，高亮当前课程
            if st.session_state.current_course == course:
                label = f"✅ {idx}. {course}"
            else:
                label = f"📖 {idx}. {course}"
            if st.button(label, key=f"sel_{course}"):
                st.session_state.current_course = course
                st.session_state.messages = []
                st.rerun()
        with c2:
            if st.button("🗑️", key=f"del_{course}"):
                delete_course(course)
                remaining = [c for c in courses if c != course]
                st.session_state.current_course = remaining[0] if remaining else None
                st.rerun()

if not st.session_state.current_course:
    st.info("👈 请先在侧边栏创建课程")
    st.stop()

course_name = st.session_state.current_course
course_dir = get_course_path(course_name)
vs = FileVectorStore(course_name)

# 默认显示问答助手（第一个标签）
tab1, tab2 = st.tabs(["💬 问答助手", "📄 课件管理"])

with tab2:
    st.markdown(f"### 当前课程：**{course_name}**")
    uploaded_files = st.file_uploader(
        "上传课件（支持 Word/TXT/PDF）",
        type=["docx", "txt", "pdf"],
        accept_multiple_files=True
    )
    if uploaded_files and st.button("上传并处理"):
        progress = st.progress(0)
        for i, file in enumerate(uploaded_files):
            file_path = course_dir / file.name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            try:
                text, ftype = parse_document(str(file_path))
                chunks = chunk_text(text)
                vs.add_documents(chunks, file.name)
                meta = {"name": file.name, "type": ftype, "chunks": len(chunks)}
                with open(course_dir / f"{file.name}.meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False)
            except Exception as e:
                st.error(f"处理 {file.name} 失败: {e}")
            progress.progress((i + 1) / len(uploaded_files))
        st.success(f"✅ 已处理 {len(uploaded_files)} 个文件，数据已持久化保存")
        st.rerun()
    
    st.markdown("#### 已上传课件")
    stats = vs.get_stats()
    st.metric("文档片段数", stats["count"])
    for f in vs.list_files():
        st.markdown(f"📄 {f}")
    st.info("💡 提示：数据已保存到文件，刷新页面不会丢失")

with tab1:
    st.markdown(f"### 💬 小书童问答 - **{course_name}**")
    st.markdown("**在下方输入框提问，最新的对话显示在最上面**")
    
    if not st.session_state.config.get("api_key"):
        st.warning("⚠️ 请先在侧边栏配置API密钥")
        st.stop()
    
    # 输入框
    question = st.chat_input("输入你的问题...")
    
    # 对话记录（最新的在上面）
    for msg in reversed(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                contexts = vs.search(question, top_k=3)
                if contexts:
                    messages = build_prompt(question, contexts, course_name)
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
        st.rerun()
    
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()
