# -*- coding: utf-8 -*-
"""课程伴学助教 - Streamlit主应用"""
import streamlit as st
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
import time
import shutil

# 导入项目模块
from config import (
    load_config, save_config, get_course_path, list_courses, 
    delete_course, MODEL_CONFIG, COURSES_DIR
)
from utils.doc_parser import parse_document, chunk_text, get_file_type_label
from utils.vector_store import get_vector_store
from utils.llm_client import LLMClient, build_rag_prompt


# 页面配置
st.set_page_config(
    page_title="课程伴学助教",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #4A90D9;
        margin-bottom: 1rem;
    }
    .course-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .doc-item {
        background: #f8f9fa;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        border-left: 3px solid #4A90D9;
    }
    .source-tag {
        background: #e8f4fd;
        color: #4A90D9;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #4A90D9;
        color: white;
    }
    .assistant-message {
        background: #f0f2f5;
        color: #333;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# 初始化session state
def init_session_state():
    """初始化会话状态"""
    if "current_course" not in st.session_state:
        st.session_state.current_course = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "config" not in st.session_state:
        st.session_state.config = load_config()
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None


def render_sidebar():
    """渲染侧边栏配置"""
    with st.sidebar:
        st.markdown("### ⚙️ 模型配置")
        
        config = st.session_state.config
        
        # 大模型选择
        provider = st.selectbox(
            "选择大模型",
            options=list(MODEL_CONFIG.keys()),
            format_func=lambda x: MODEL_CONFIG[x]["name"],
            index=list(MODEL_CONFIG.keys()).index(config.get("provider", "deepseek")),
            key="provider_select"
        )
        
        # API Key
        api_key = st.text_input(
            "API Key",
            value=config.get("api_key", ""),
            type="password",
            help="输入你的API密钥"
        )
        
        # 模型选择
        model_options = MODEL_CONFIG[provider]["models"]
        default_model = config.get("model", MODEL_CONFIG[provider]["default_model"])
        if default_model not in model_options:
            default_model = model_options[0]
        
        model = st.selectbox(
            "选择模型",
            options=model_options,
            index=model_options.index(default_model) if default_model in model_options else 0,
            key="model_select"
        )
        
        # 保存按钮
        if st.button("💾 保存配置", type="primary"):
            new_config = {
                "provider": provider,
                "api_key": api_key,
                "model": model,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            }
            if save_config(new_config):
                st.session_state.config = new_config
                st.session_state.llm_client = None  # 重置客户端
                st.success("✅ 配置已保存！")
            else:
                st.error("❌ 保存失败")
        
        # 测试连接
        if api_key:
            if st.button("🔗 测试连接"):
                with st.spinner("测试中..."):
                    success, msg = LLMClient.test_connection(provider, api_key, model)
                    if success:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")
        
        st.divider()
        
        # 课程管理
        st.markdown("### 📚 课程管理")
        
        # 创建新课程
        new_course = st.text_input("新课程名称", placeholder="输入课程名称...")
        if st.button("➕ 创建课程") and new_course:
            if new_course not in list_courses():
                course_dir = get_course_path(new_course)
                st.session_state.current_course = new_course
                st.success(f"✅ 课程「{new_course}」已创建！")
                st.rerun()
            else:
                st.warning("⚠️ 课程已存在")
        
        # 课程列表
        courses = list_courses()
        if courses:
            st.markdown("**已有课程：**")
            for course in courses:
                col1, col2 = st.columns([3, 1])
                is_active = st.session_state.current_course == course
                with col1:
                    if st.button(
                        f"📖 {course}",
                        key=f"course_{course}",
                        type="primary" if is_active else "secondary"
                    ):
                        st.session_state.current_course = course
                        st.session_state.messages = []  # 切换课程时清空对话
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{course}"):
                        if delete_course(course):
                            if st.session_state.current_course == course:
                                st.session_state.current_course = None
                            st.success("已删除")
                            st.rerun()
                        else:
                            st.error("删除失败")


def render_course_page():
    """渲染课程管理页面"""
    st.markdown('<p class="main-header">📚 课程管理</p>', unsafe_allow_html=True)
    
    if not st.session_state.current_course:
        st.info("👈 请先在侧边栏选择或创建课程")
        return
    
    course_name = st.session_state.current_course
    course_dir = get_course_path(course_name)
    
    st.markdown(f"### 当前课程：**{course_name}**")
    
    # 标签页：课件管理 / 向量状态
    tab1, tab2 = st.tabs(["📄 课件管理", "📊 向量状态"])
    
    with tab1:
        # 上传课件
        st.markdown("#### 上传课件")
        uploaded_files = st.file_uploader(
            "选择文件",
            type=["docx", "txt", "pdf", "mp3", "wav", "m4a"],
            accept_multiple_files=True,
            help="支持 Word、文本、PDF、音频文件"
        )
        
        if uploaded_files:
            if st.button("📤 开始上传并处理", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"处理: {file.name}")
                    
                    # 保存文件
                    file_path = course_dir / file.name
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    try:
                        # 解析文档
                        with st.spinner("解析文档..."):
                            text, file_type = parse_document(str(file_path))
                        
                        # 切分文本
                        with st.spinner("切分内容..."):
                            chunks = chunk_text(text)
                        
                        # 存储到向量库
                        with st.spinner("存储向量..."):
                            try:
                                vs = get_vector_store(course_name, str(COURSES_DIR.parent))
                                vs.add_documents(course_name, chunks, file.name)
                            except Exception as e:
                                st.warning(f"向量存储失败: {e}")
                        
                        # 保存元数据
                        meta_file = course_dir / f"{file.name}.meta.json"
                        with open(meta_file, "w", encoding="utf-8") as f:
                            json.dump({
                                "name": file.name,
                                "type": file_type,
                                "size": file.size,
                                "upload_time": datetime.now().isoformat(),
                                "chunks": len(chunks)
                            }, f, ensure_ascii=False, indent=2)
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                    except Exception as e:
                        st.error(f"处理 {file.name} 失败: {e}")
                
                status_text.text("完成！")
                st.success(f"✅ 已成功处理 {len(uploaded_files)} 个文件")
                st.rerun()
        
        # 课件列表
        st.markdown("#### 课件列表")
        
        meta_files = list(course_dir.glob("*.meta.json"))
        if meta_files:
            for meta_file in meta_files:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    file_icon = {
                        "word": "📝", "txt": "📃", "pdf": "📕", "audio": "🎧"
                    }.get(meta["type"], "📄")
                    st.markdown(f"{file_icon} **{meta['name']}**")
                    st.caption(f"类型: {get_file_type_label(Path(meta['name']).suffix)} | "
                              f"切片: {meta.get('chunks', '?')}个 | "
                              f"上传: {meta.get('upload_time', '未知')[:19]}")
                with col2:
                    size_mb = meta.get('size', 0) / (1024 * 1024)
                    st.caption(f"{size_mb:.2f} MB")
                with col3:
                    if st.button("🗑️", key=f"del_{meta['name']}"):
                        # 删除文件和元数据
                        file_to_del = course_dir / meta['name']
                        if file_to_del.exists():
                            file_to_del.unlink()
                        meta_file.unlink()
                        # 删除向量
                        try:
                            vs = get_vector_store(course_name, str(COURSES_DIR.parent))
                            vs.delete_file(course_name, meta['name'])
                        except:
                            pass
                        st.rerun()
        else:
            st.info("暂无课件，请上传文件")
    
    with tab2:
        st.markdown("#### 向量数据库状态")
        try:
            vs = get_vector_store(course_name, str(COURSES_DIR.parent))
            info = vs.get_collection_info(course_name)
            if info:
                st.metric("文档片段数", info.get("count", 0))
                files = vs.list_all_files(course_name)
                st.markdown(f"**已索引文件:** {len(files)} 个")
                for f in files:
                    st.markdown(f"- {f}")
            else:
                st.info("暂无向量数据")
        except Exception as e:
            st.error(f"获取状态失败: {e}")


def render_chat_page():
    """渲染问答助手页面"""
    st.markdown('<p class="main-header">💬 课程问答助手</p>', unsafe_allow_html=True)
    
    # 检查配置
    config = st.session_state.config
    if not config.get("api_key"):
        st.warning("⚠️ 请先在侧边栏配置API Key")
        return
    
    if not st.session_state.current_course:
        st.info("👈 请先在侧边栏选择课程")
        return
    
    course_name = st.session_state.current_course
    course_dir = get_course_path(course_name)
    
    # 检查是否有课件
    meta_files = list(course_dir.glob("*.meta.json"))
    if not meta_files:
        st.info("📤 该课程暂无课件，请先上传课件")
        return
    
    # 显示当前课程
    st.markdown(f"**当前课程:** 📚 {course_name}")
    
    # 清空对话按钮
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            st.rerun()
    
    # 显示对话历史
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
    
    # 用户输入
    if prompt := st.chat_input("输入你的问题..."):
        # 添加用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        with st.chat_message("user"):
            st.write(prompt)
        
        # 生成AI回复
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    # 初始化LLM客户端
                    if st.session_state.llm_client is None:
                        st.session_state.llm_client = LLMClient(
                            provider=config["provider"],
                            api_key=config["api_key"],
                            model=config["model"]
                        )
                    
                    # 初始化向量存储
                    vs = get_vector_store(course_name, str(COURSES_DIR.parent))
                    
                    # 检索相关文档
                    relevant_docs = vs.search(course_name, prompt, top_k=3)
                    
                    if not relevant_docs:
                        response = "抱歉，我在课程课件中没有找到相关内容。请尝试重新上传课件或调整问题。"
                        st.write(response)
                    else:
                        # 构建RAG提示词
                        messages = build_rag_prompt(prompt, relevant_docs, course_name)
                        
                        # 调用LLM
                        response = st.session_state.llm_client.chat(messages)
                        st.write(response)
                    
                    # 保存助手消息
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                except Exception as e:
                    error_msg = f"❌ 处理失败: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def render_welcome():
    """渲染欢迎页面"""
    st.markdown("""
    # 📚 课程伴学助教
    
    基于AI大模型的课程学习助手，帮助你更好地学习和理解课程内容。
    
    ## ✨ 功能特点
    
    ### 1. 多格式支持
    - 📝 Word文档 (.docx)
    - 📃 文本文件 (.txt)  
    - 📕 PDF文档 (.pdf)
    - 🎧 音频文件 (.mp3, .wav, .m4a)
    
    ### 2. 智能问答
    - 基于RAG技术，从课件内容中检索答案
    - 标注答案来源，方便追溯学习
    
    ### 3. 多模型支持
    - DeepSeek
    - 通义千问
    - OpenAI GPT系列
    
    ---
    
    ## 🚀 快速开始
    
    1. **配置API**: 在左侧边栏填写你的API Key
    2. **创建课程**: 输入课程名称并创建
    3. **上传课件**: 上传你的课程材料
    4. **开始问答**: 选择课程后即可提问！
    
    ---
    
    """)
    
    # 显示示例
    st.markdown("### 📖 使用示例")
    
    st.info("📌 **答案来源标注示例**")
    st.markdown("""
    > 当你问："AI有什么特点？"
    
    AI回答后，会在末尾标注：
    ```
    📖 来源: 【快速入门】认识你的新伙伴 - 一、AI到底是什么？别怕，它就是个超级实习生
    ```
    """)


def main():
    """主函数"""
    init_session_state()
    render_sidebar()
    
    # 主页面选项卡
    page = st.sidebar.radio(
        "功能导航",
        ["🏠 首页", "📚 课程管理", "💬 问答助手"],
        index=0
    )
    
    if page == "🏠 首页":
        render_welcome()
    elif page == "📚 课程管理":
        render_course_page()
    elif page == "💬 问答助手":
        render_chat_page()


if __name__ == "__main__":
    main()
