# -*- coding: utf-8 -*-
"""大模型调用模块 - 支持 DeepSeek/通义千问/OpenAI"""
import os
from typing import List, Dict, Optional, Tuple
from config import MODEL_CONFIG


class LLMClient:
    """大模型调用客户端"""
    
    def __init__(self, provider: str, api_key: str, model: str = None):
        """
        初始化LLM客户端
        
        Args:
            provider: 提供商 (deepseek/qwen/openai)
            api_key: API密钥
            model: 模型名称
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model or MODEL_CONFIG[provider]["default_model"]
        self.api_base = MODEL_CONFIG[provider]["api_base"]
        
        self._client = None
    
    def _get_client(self):
        """获取API客户端"""
        if self._client is None:
            if self.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base
                )
            elif self.provider == "deepseek":
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base
                )
            elif self.provider == "qwen":
                import dashscope
                dashscope.api_key = self.api_key
                self._client = "dashscope"
        
        return self._client
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        发送对话请求
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数
        
        Returns:
            str: 助手回复
        """
        client = self._get_client()
        
        if client == "dashscope":
            return self._chat_dashscope(messages, temperature, max_tokens)
        else:
            return self._chat_openai_compatible(client, messages, temperature, max_tokens)
    
    def _chat_openai_compatible(self, client, messages, temperature, max_tokens) -> str:
        """调用OpenAI兼容接口"""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"API调用失败: {e}")
    
    def _chat_dashscope(self, messages, temperature, max_tokens) -> str:
        """调用阿里云DashScope接口"""
        try:
            from dashscope import Generation
            
            # 转换消息格式
            messages_dashscope = []
            for msg in messages:
                role = msg["role"]
                if role == "assistant":
                    role = "assistant"
                elif role == "system":
                    role = "system"
                else:
                    role = "user"
                messages_dashscope.append({
                    "role": role,
                    "content": msg["content"]
                })
            
            response = Generation.call(
                model=self.model,
                messages=messages_dashscope,
                temperature=temperature,
                max_tokens=max_tokens,
                result_format="message"
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                raise Exception(f"DashScope API错误: {response.message}")
                
        except Exception as e:
            raise Exception(f"DashScope API调用失败: {e}")
    
    @staticmethod
    def test_connection(provider: str, api_key: str, model: str) -> Tuple[bool, str]:
        """
        测试API连接
        
        Returns:
            (是否成功, 消息)
        """
        try:
            client = LLMClient(provider, api_key, model)
            # 发送一个简单的测试请求
            response = client.chat([
                {"role": "user", "content": "你好，请回复'连接成功'。"}
            ], max_tokens=50)
            
            if "成功" in response or "OK" in response or "你好" in response:
                return True, "连接成功"
            else:
                return True, f"连接正常，响应: {response[:50]}"
                
        except Exception as e:
            return False, str(e)


def build_rag_prompt(query: str, context_docs: List[Dict], course_name: str = "") -> List[Dict]:
    """
    构建RAG提示词
    
    Args:
        query: 用户问题
        context_docs: 检索到的相关文档
        course_name: 课程名称
    
    Returns:
        List[Dict]: 消息列表
    """
    # 构建上下文
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        section = doc.get("metadata", {}).get("section", "")
        source_info = f"📖 来源: {course_name}"
        if section:
            source_info += f" - {section}"
        
        context_parts.append(f"【参考内容 {i}】\n{doc['text']}\n{source_info}")
    
    context_text = "\n\n".join(context_parts)
    
    system_prompt = f"""你是一个课程伴学助手，基于提供的课件内容回答学生的问题。

重要规则：
1. 回答必须基于提供的参考内容，不要编造信息
2. 如果参考内容中没有相关信息，请明确告知学生
3. 在回答末尾，标注答案来源（使用 📖 来源: 课程名 - 章节名 格式）
4. 回答要清晰、有条理，使用学生易懂的语言
5. 如果需要，可以对参考内容进行总结和解释

当前课程：{course_name}

参考内容：
{context_text}
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    return messages


def build_summary_prompt(document_text: str, file_name: str) -> List[Dict]:
    """
    构建文档摘要提示词（用于提取文档大纲）
    """
    system_prompt = """你是一个课程内容分析助手。请分析提供的课件内容，提取以下信息：

1. 文档标题
2. 主要章节列表（用数字或标题列出）
3. 每个章节的简要说明

请用简洁的方式呈现结果。
"""
    
    # 截取前2000字进行分析
    sample_text = document_text[:2000] + "..." if len(document_text) > 2000 else document_text
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请分析以下课件内容（文件: {file_name}）：\n\n{sample_text}"}
    ]
    
    return messages
