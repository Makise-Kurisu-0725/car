import os
import json
import uuid
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core._cancellation_token import CancellationToken
from autogen_agentchat.messages import TextMessage

class DictMemoryConfig(BaseModel):
    """
    字典存储记忆的配置类。

    该类用于配置字典存储记忆的相关参数，包括持久化路径、最大记忆项数量、查询匹配方法以及查询时返回的最相似结果数量。

    核心功能:
    - 配置字典记忆的持久化路径。
    - 设置最大记忆项数量，超过此数量将移除最早的记忆。
    - 选择查询匹配方法：'substring'（子字符串匹配）、'exact'（精确匹配）、'keyword'（关键词匹配）。
    - 设置查询时返回的最相似结果数量。

    使用示例:

    构造函数参数:
    - path (Optional[str]): 字典记忆持久化的保存路径。默认为 None。
    - max_items (int): 最大记忆项数量，超过此数量将移除最早的记忆。默认为 1000。
    - query_match_method (str): 查询匹配方法。可选值：'substring'（子字符串）、'exact'（精确）、'keyword'（关键词）。默认为 'substring'。
    - top_k (int): 查询时返回的最相似结果数量。默认为 5。

    特殊使用限制或潜在的副作用:
    - 如果 `path` 为 None，则记忆不会持久化保存。
    - `max_items` 应该大于 0，否则可能导致无法存储任何记忆。
    - `query_match_method` 的选择会影响查询的效率和准确性。
    - `top_k` 的值应该根据实际需求设置，过大的值可能会影响查询性能。
    """
    
    path: Optional[str] = Field(
        default=None, 
        description="字典记忆持久化的保存路径"
    )
    max_items: int = Field(
        default=1000, 
        description="最大记忆项数量，超过此数量将移除最早的记忆"
    )
    query_match_method: str = Field(
        default="substring", 
        description="查询匹配方法: 'substring'(子字符串), 'exact'(精确), 'keyword'(关键词)"
    )
    top_k: int = Field(
        default=5, 
        description="查询时返回的最相似结果数量"
    )


class DictMemory(Memory):
    """
    基于字典的记忆实现，用于简单键值对形式的记忆内容存储。
    支持基于关键词和子字符串的文本内容搜索。

    核心功能：
    - 存储和检索记忆内容
    - 更新模型上下文中的相关记忆
    - 支持从文件加载和保存记忆

    使用示例：

    构造函数参数：
    - config: 配置参数，类型为DictMemoryConfig，包含路径、最大记忆数量等配置信息。

    特殊使用限制或潜在的副作用：
    - 如果配置中设置了路径，会在初始化时尝试从文件加载记忆，并在关闭时保存记忆。
    - 如果记忆数量超过最大限制，会自动移除最早的记忆。
    - 在保存和加载记忆时，可能会出现文件读写错误。
    """
    
    def __init__(self, config: DictMemoryConfig = DictMemoryConfig()):
        """
        初始化字典记忆
        
        Args:
            config: 配置参数
        """
        super().__init__()
        self.config = config
        
        # 初始化内存字典和记忆列表
        self.memories = {}  # 键为记忆ID，值为MemoryContent
        self.memory_keys = []  # 按时间顺序记录ID，用于实现FIFO
        
        # 如果设置了路径，尝试从文件加载记忆
        if config.path and os.path.exists(config.path):
            self._load_from_file()
    
    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """
        使用相关记忆内容更新提供的模型上下文
        
        Args:
            model_context: 要更新的上下文
            
        Returns:
            UpdateContextResult 包含相关记忆
        """
        # 从上下文中提取最后一条消息作为查询内容
        query_text = ""
        if model_context._messages and len(model_context._messages) > 0:
            last_message = model_context._messages[-1]
            if isinstance(last_message.content, str):
                query_text = last_message.content
            # 如果是列表格式，尝试提取文本内容
            elif isinstance(last_message.content, list):
                for content_item in last_message.content:
                    if isinstance(content_item, dict) and content_item.get("type") == "text":
                        query_text = content_item.get("text", "")
                        if query_text:
                            break
        
        if not query_text:
            # 如果没有找到合适的查询内容，返回空结果
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))
        
        # 查询相关记忆
        query_content = MemoryContent(
            content=query_text,
            mime_type=MemoryMimeType.TEXT
        )
        memory_results = await self.query(query_content)
        
        # 如果有相关记忆，将其添加到上下文最后
        if memory_results.results:
            memory_texts = []
            for mem in memory_results.results:
                if isinstance(mem.content, str):
                    memory_texts.append(mem.content)
            
            if memory_texts:
                memory_content = "相关记忆信息:\n" + "\n---\n".join(memory_texts)
                
                # 将记忆内容添加到上下文最后
                model_context._messages.append(
                    TextMessage(
                        source="DictMemory",
                        content=memory_content,
                        type="TextMessage"
                    )
                )
        
        return UpdateContextResult(memories=memory_results)
    
    async def query(
        self,
        query: Union[str, MemoryContent],
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """
        查询记忆库并返回相关条目
        
        Args:
            query: 查询内容
            cancellation_token: 可选的取消操作令牌
            **kwargs: 额外的实现特定参数
            
        Returns:
            MemoryQueryResult 包含记忆条目
        """
        # 检查是否已被取消
        if cancellation_token and cancellation_token._cancelled:
            return MemoryQueryResult(results=[])
        
        # 提取查询文本
        query_text = ""
        if isinstance(query, str):
            query_text = query
        elif isinstance(query, MemoryContent):
            if isinstance(query.content, str):
                query_text = query.content
            elif isinstance(query.content, dict) and "text" in query.content:
                query_text = query.content["text"]
        
        if not query_text:
            return MemoryQueryResult(results=[])
        
        # 设置top_k参数
        top_k = kwargs.get("top_k", self.config.top_k)
        match_method = kwargs.get("match_method", self.config.query_match_method)
        
        # 根据匹配方法过滤记忆
        matched_memories = []
        query_text_lower = query_text.lower()
        
        for memory_id in self.memory_keys:
            memory = self.memories.get(memory_id)
            if not memory:
                continue
                
            # 只处理文本内容
            if not isinstance(memory.content, str):
                continue
                
            content_lower = memory.content.lower()
            matched = False
            score = 0.0
            
            if match_method == "exact":
                # 精确匹配
                if query_text_lower == content_lower:
                    matched = True
                    score = 1.0
            elif match_method == "substring":
                # 子字符串匹配
                if query_text_lower in content_lower or content_lower in query_text_lower:
                    matched = True
                    # 简单的相似度计算 - 共同子串长度比例
                    common_length = len(min(query_text_lower, content_lower))
                    max_length = len(max(query_text_lower, content_lower))
                    score = common_length / max_length if max_length > 0 else 0
            elif match_method == "keyword":
                # 关键词匹配
                query_words = set(query_text_lower.split())
                content_words = set(content_lower.split())
                common_words = query_words.intersection(content_words)
                
                if common_words:
                    matched = True
                    # 计算关键词匹配率
                    score = len(common_words) / max(len(query_words), len(content_words))
            
            if matched:
                matched_memories.append((memory, score))
        
        # 按相似度排序并限制数量
        matched_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = matched_memories[:top_k]
        
        # 构建结果
        results = []
        for memory, score in top_memories:
            # 添加相似度分数到元数据
            metadata = memory.metadata or {}
            metadata["score"] = score
            
            # 创建新的MemoryContent，包含原始内容和更新的元数据
            result_memory = MemoryContent(
                content=memory.content,
                mime_type=memory.mime_type,
                metadata=metadata
            )
            results.append(result_memory)
        
        return MemoryQueryResult(results=results)
    
    async def add(self, content: MemoryContent, cancellation_token: Optional[CancellationToken] = None) -> None:
        """
        向记忆库添加新内容
        
        Args:
            content: 要添加的记忆内容
            cancellation_token: 可选的取消操作令牌
        """
        # 检查是否已被取消
        if cancellation_token and cancellation_token._cancelled:
            return
        
        # 生成唯一ID
        memory_id = str(uuid.uuid4())
        
        # 存储记忆内容
        self.memories[memory_id] = content
        self.memory_keys.append(memory_id)
        
        # 如果超过最大记忆数量，移除最早的记忆
        while len(self.memory_keys) > self.config.max_items:
            oldest_id = self.memory_keys.pop(0)
            if oldest_id in self.memories:
                del self.memories[oldest_id]
        
        # 如果设置了路径，保存到文件
        if self.config.path:
            await self._save_to_file()
    
    async def clear(self) -> None:
        """
        清除所有记忆条目
        """
        self.memories = {}
        self.memory_keys = []
        
        # 如果设置了路径，删除文件
        if self.config.path and os.path.exists(self.config.path):
            try:
                os.remove(self.config.path)
            except Exception as e:
                print(f"清除记忆文件时出错: {e}")
    
    async def close(self) -> None:
        """
        清理记忆实现使用的资源
        """
        # 如果设置了路径，确保在关闭前保存数据
        if self.config.path:
            await self._save_to_file()
    
    async def _save_to_file(self) -> None:
        """将记忆保存到文件"""
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(os.path.abspath(self.config.path)), exist_ok=True)
            
            # 准备要保存的数据
            serializable_memories = {}
            for memory_id, memory in self.memories.items():
                # 序列化MemoryContent
                memory_dict = {
                    "content": memory.content if isinstance(memory.content, (str, bytes)) else json.dumps(memory.content),
                    "mime_type": memory.mime_type.value if isinstance(memory.mime_type, MemoryMimeType) else memory.mime_type,
                    "metadata": memory.metadata
                }
                serializable_memories[memory_id] = memory_dict
            
            save_data = {
                "memories": serializable_memories,
                "memory_keys": self.memory_keys
            }
            
            # 以JSON格式保存
            with open(self.config.path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆到文件时出错: {e}")
    
    def _load_from_file(self) -> None:
        """从文件加载记忆"""
        try:
            with open(self.config.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载记忆键顺序
            self.memory_keys = data.get("memory_keys", [])
            
            # 加载记忆内容
            serialized_memories = data.get("memories", {})
            for memory_id, memory_dict in serialized_memories.items():
                # 尝试解析内容
                content = memory_dict.get("content", "")
                mime_type_str = memory_dict.get("mime_type", "text/plain")
                metadata = memory_dict.get("metadata")
                
                # 将MIME类型字符串转换为枚举
                try:
                    mime_type = MemoryMimeType(mime_type_str)
                except ValueError:
                    mime_type = mime_type_str
                
                # 创建MemoryContent对象
                memory = MemoryContent(
                    content=content,
                    mime_type=mime_type,
                    metadata=metadata
                )
                self.memories[memory_id] = memory
        except Exception as e:
            print(f"从文件加载记忆时出错: {e}")
            # 初始化空记忆
            self.memories = {}
            self.memory_keys = []
