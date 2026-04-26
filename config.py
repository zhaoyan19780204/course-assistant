# -*- coding: utf-8 -*-
"""配置管理模块"""
import os
import json
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = DATA_DIR / "config.json"
COURSES_DIR = DATA_DIR / "courses"

# 确保目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)
COURSES_DIR.mkdir(parents=True, exist_ok=True)

# 默认配置
DEFAULT_CONFIG = {
    "provider": "deepseek",
    "api_key": "",
    "model": "deepseek-chat",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}

# 支持的模型配置
MODEL_CONFIG = {
    "deepseek": {
        "name": "DeepSeek",
        "models": ["deepseek-chat", "deepseek-coder"],
        "api_base": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat"
    },
    "qwen": {
        "name": "通义千问",
        "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen-max-longcontext"],
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus"
    },
    "openai": {
        "name": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "api_base": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini"
    }
}


def load_config() -> dict:
    """加载配置"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_config(config: dict) -> bool:
    """保存配置"""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存配置失败: {e}")
        return False


def get_course_path(course_name: str) -> Path:
    """获取课程数据目录"""
    course_dir = COURSES_DIR / course_name
    course_dir.mkdir(parents=True, exist_ok=True)
    return course_dir


def list_courses() -> list:
    """列出所有课程"""
    if not COURSES_DIR.exists():
        return []
    return [d.name for d in COURSES_DIR.iterdir() if d.is_dir()]


def delete_course(course_name: str) -> bool:
    """删除课程"""
    import shutil
    course_dir = COURSES_DIR / course_name
    if course_dir.exists():
        try:
            shutil.rmtree(course_dir)
            return True
        except Exception:
            return False
    return False
