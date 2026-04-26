# -*- coding: utf-8 -*-
"""文档解析模块 - 支持 Word/TXT/PDF/音频"""
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import tempfile

# 文档解析
def parse_docx(file_path: str) -> str:
    """解析Word文档"""
    try:
        from docx import Document
        doc = Document(file_path)
        texts = []
        for para in doc.paragraphs:
            if para.text.strip():
                texts.append(para.text.strip())
        return "\n".join(texts)
    except Exception as e:
        raise Exception(f"Word文档解析失败: {e}")


def parse_txt(file_path: str) -> str:
    """解析TXT文档"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise Exception("TXT文档编码不受支持")


def parse_pdf(file_path: str) -> str:
    """解析PDF文档"""
    try:
        import pdfplumber
        texts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)
        if not texts:
            raise Exception("PDF中未提取到文本")
        return "\n".join(texts)
    except Exception as e:
        raise Exception(f"PDF文档解析失败: {e}")


def parse_audio(file_path: str) -> str:
    """解析音频文件，转写为文字"""
    raise Exception("音频转写功能暂不可用（需要额外配置）。请上传 Word/TXT/PDF 格式的课件。")


def parse_document(file_path: str) -> Tuple[str, str]:
    """
    解析任意支持的文档
    返回: (提取的文本, 文件类型)
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == ".docx":
        return parse_docx(file_path), "word"
    elif suffix == ".txt":
        return parse_txt(file_path), "txt"
    elif suffix == ".pdf":
        return parse_pdf(file_path), "pdf"
    elif suffix in [".mp3", ".wav", ".m4a", ".ogg", ".flac"]:
        return parse_audio(file_path), "audio"
    else:
        raise Exception(f"不支持的文件格式: {suffix}")


def _is_section_heading(line: str) -> Tuple[bool, str]:
    """
    检测是否为章节标题
    
    Returns:
        (is_heading, heading_type)
        heading_type: "main" (主章节) / "subsection" (子章节) / "" (非标题)
    """
    line = line.strip()
    if not line or len(line) > 100:
        return False, ""
    
    # 主章节标题模式
    main_patterns = [
        r'^[一二三四五六七八九十]+、',  # 一、...
        r'^第[一二三四五六七八九十百千万]+[章节篇部]',  # 第一章、第一节
        r'^【[^】]+】',  # 【标题】
        r'^[A-Z][A-Z0-9\s]+$',  # 全大写标题
        r'^《[^》]+》',  # 《书名》
    ]
    
    for pattern in main_patterns:
        if re.match(pattern, line):
            return True, "main"
    
    # 子章节模式（短标题，通常有编号）
    subsection_patterns = [
        r'^\d{1,2}\s+\S+',  # 01 前言、1. 概述
        r'^[0-9]+\.\s*\S+',  # 1. 前言
        r'^[0-9]+[a-z]\s+\S+',  # 1a 内容
        r'^[A-Z]\.\s*\S+',  # A. 概述
    ]
    
    for pattern in subsection_patterns:
        if re.match(pattern, line) and len(line) < 40:
            return True, "subsection"
    
    return False, ""


def _is_short_line(line: str) -> bool:
    """判断是否为短行（如目录项）"""
    line = line.strip()
    if not line:
        return False
    # 纯数字开头 + 短文本
    if re.match(r'^\d+[\s\.、]', line) and len(line) < 30:
        return True
    return False


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 30) -> List[Dict]:
    """
    将文本按章节和段落智能切分
    
    Args:
        text: 原始文本
        chunk_size: 每个chunk的目标字数
        overlap: 相邻chunk的重叠字数
    
    Returns:
        List[Dict]: [{"text": "...", "chunk_id": 0, "metadata": {...}}, ...]
    """
    chunks = []
    
    lines = text.split('\n')
    
    current_main_section = ""  # 当前主章节
    current_sub_section = ""   # 当前子章节
    current_content = []       # 当前积累的内容
    current_chars = 0
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 跳过空行
        if not line:
            i += 1
            continue
        
        # 检测标题
        is_heading, heading_type = _is_section_heading(line)
        
        # 跳过目录行（短行通常为目录）
        is_toc_line = _is_short_line(line) and current_chars == 0
        
        if is_heading and not is_toc_line:
            # 保存之前的chunk
            if current_content and current_chars > 0:
                chunk_text_content = "\n".join(current_content)
                # 跳过太短的chunk
                if len(chunk_text_content) > 50:
                    chunks.append({
                        "text": chunk_text_content,
                        "chunk_id": len(chunks),
                        "metadata": {
                            "main_section": current_main_section,
                            "sub_section": current_sub_section
                        }
                    })
            
            # 更新章节
            if heading_type == "main":
                current_main_section = line
                current_sub_section = ""
            else:
                current_sub_section = line
            
            current_content = []
            current_chars = 0
            
        elif is_toc_line:
            # 目录行，跳过
            i += 1
            continue
        
        else:
            # 内容行
            # 如果当前内容超长，先保存一部分
            if current_chars + len(line) > chunk_size and current_content:
                chunk_text_content = "\n".join(current_content)
                chunks.append({
                    "text": chunk_text_content,
                    "chunk_id": len(chunks),
                    "metadata": {
                        "main_section": current_main_section,
                        "sub_section": current_sub_section
                    }
                })
                
                # 保留最后几行作为overlap
                overlap_count = min(3, len(current_content))
                current_content = current_content[-overlap_count:]
                current_chars = sum(len(l) for l in current_content)
            
            current_content.append(line)
            current_chars += len(line)
        
        i += 1
    
    # 处理最后一个chunk
    if current_content and current_chars > 50:
        chunks.append({
            "text": "\n".join(current_content),
            "chunk_id": len(chunks),
            "metadata": {
                "main_section": current_main_section,
                "sub_section": current_sub_section
            }
        })
    
    return chunks


def get_file_type_label(suffix: str) -> str:
    """获取文件类型标签"""
    type_labels = {
        ".docx": "Word文档",
        ".txt": "文本文件",
        ".pdf": "PDF文档",
        ".mp3": "音频(MP3)",
        ".wav": "音频(WAV)",
        ".m4a": "音频(M4A)",
        ".ogg": "音频(OGG)",
        ".flac": "音频(FLAC)"
    }
    return type_labels.get(suffix.lower(), "未知格式")


class DocumentParser:
    """文档解析器"""
    
    @staticmethod
    def parse(file_path: str) -> Tuple[str, str]:
        """解析文档，返回 (文本内容, 文件类型)"""
        return parse_document(file_path)
    
    @staticmethod
    def chunk(text: str, chunk_size: int = 600, overlap: int = 30) -> List[Dict]:
        """切分文本"""
        return chunk_text(text, chunk_size, overlap)
    
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """获取文件类型"""
        suffix = Path(file_path).suffix.lower()
        return get_file_type_label(suffix)
