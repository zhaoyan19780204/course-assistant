# 📚 课程伴学助教

基于AI大模型的课程学习助手，支持上传课件（Word/TXT/PDF/音频），AI基于课件内容回答问题，并标注答案来源。

## ✨ 功能特点

### 1. 多格式文档支持
- 📝 Word文档 (.docx)
- 📃 文本文件 (.txt)
- 📕 PDF文档 (.pdf)
- 🎧 音频文件 (.mp3, .wav, .m4a) - 自动转写为文字

### 2. 智能问答
- 基于RAG（检索增强生成）技术
- 从课件内容中检索相关信息
- **标注答案来源**，方便追溯学习

### 3. 多模型支持
- DeepSeek
- 通义千问
- OpenAI GPT系列

## 🚀 快速开始

### 本地运行

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install streamlit python-docx pdfplumber chromadb openai-whisper openai dashscope sentence-transformers numpy
```

#### 2. 配置API

首次运行时会自动创建配置文件。也可以手动创建 `data/config.json`：

```json
{
    "provider": "deepseek",
    "api_key": "your-api-key-here",
    "model": "deepseek-chat",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

#### 3. 运行应用

```bash
streamlit run app.py
```

应用默认运行在 `http://localhost:8501`

## 📁 项目结构

```
course-assistant/
├── app.py                 # Streamlit主应用
├── config.py              # 配置管理
├── requirements.txt       # 依赖清单
├── README.md              # 说明文档
├── utils/
│   ├── __init__.py
│   ├── doc_parser.py      # 文档解析（Word/TXT/PDF/音频）
│   ├── vector_store.py    # 向量存储（ChromaDB）
│   └── llm_client.py      # 大模型调用
└── data/
    ├── config.json        # 用户配置（自动生成）
    └── courses/           # 课程数据（自动创建）
        └── [课程名]/
            ├── 课件文件...
            └── 向量数据库
```

## ☁️ Streamlit Cloud 部署

### 方法一：从GitHub部署

1. 将代码推送到GitHub仓库
2. 访问 [Streamlit Cloud](https://streamlit.io/cloud)
3. 点击 "New app"
4. 选择你的GitHub仓库和分支
5. 设置主文件路径为 `app.py`
6. 点击 "Deploy!"

### 方法二：使用requirements.txt

确保项目根目录包含：
- `app.py`
- `config.py`
- `requirements.txt`
- `utils/` 目录

### 注意事项

1. **API Key安全**：部署到云端后，API Key会暴露。建议：
   - 使用环境变量存储API Key
   - 部署到私有服务器

2. **向量数据库**：ChromaDB使用本地存储，云端部署需要持久化存储：
   - Streamlit Cloud支持持久化存储
   - 或使用其他云数据库（如Pinecone）

3. **音频转写**：whisper需要下载模型，首次运行会较慢

### 环境变量配置（可选）

在Streamlit Cloud的Secrets中配置：

```toml
[secrets]
DEEPSEEK_API_KEY = "your-key"
```

然后修改代码读取环境变量。

## 📖 使用流程

### 1. 配置模型
- 在侧边栏选择大模型提供商
- 输入API Key
- 选择具体模型
- 点击"保存配置"

### 2. 创建课程
- 在侧边栏输入新课程名称
- 点击"创建课程"

### 3. 上传课件
- 进入"课程管理"页面
- 选择文件上传
- 支持批量上传
- 等待处理完成

### 4. 开始问答
- 进入"问答助手"页面
- 输入问题
- AI基于课件内容回答，并标注来源

## 🔧 技术栈

- **前端**: Streamlit
- **文档解析**: python-docx, pdfplumber, whisper
- **向量数据库**: ChromaDB
- **Embedding**: sentence-transformers
- **大模型**: OpenAI兼容API (DeepSeek/通义千问/GPT)

## 📝 答案来源标注示例

当用户提问时，AI回答末尾会标注来源：

```
根据课件内容，AI的特点包括：

1. **知识丰富**：它读过整个互联网，拥有全人类的公开知识
2. **反应迅速**：几秒钟就能完成回答
3. **多才多艺**：可以回答问题、写作、编程

📖 来源: 【快速入门】认识你的新伙伴 - 一、AI到底是什么？别怕，它就是个超级实习生
```

## ⚠️ 注意事项

1. 音频转写依赖whisper，需要足够内存
2. 首次使用会下载embedding模型
3. 建议单课程上传不超过50个文件
4. 答案准确性取决于课件质量和问题清晰度

## 📄 License

MIT License
