import os
import asyncio
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core._cancellation_token import CancellationToken
from autogen_agentchat.messages import TextMessage
from .utils import VectorDB


class VectorFaissMemoryConfig(BaseModel):
    """
    FAISS向量存储记忆的配置类。

    该类用于配置FAISS向量存储记忆的相关参数，包括向量数据库的保存路径、文本嵌入模型、嵌入向量的维度、查询时返回的最相似结果数量、相似度阈值、相似度度量方式以及嵌入API的密钥等。

    核心功能:
    - 配置向量数据库的持久化保存路径。
    - 设置用于文本嵌入的模型名称。
    - 指定嵌入向量的维度。
    - 定义查询时返回的最相似结果数量。
    - 设置相似度阈值，过滤低于此值的结果。
    - 选择相似度度量方式（欧氏距离、余弦相似度、内积）。
    - 提供嵌入API的密钥（可选）。

    使用示例:

    构造函数参数:
    - path (Optional[str]): 向量数据库持久化的保存路径。默认为None。
    - embedding_model (str): 用于文本嵌入的模型名称。默认为"BAAI/bge-m3"。
    - dimension (int): 嵌入向量的维度。默认为512。
    - top_k (int): 查询时返回的最相似结果数量。默认为5。
    - similarity_threshold (float): 相似度阈值，低于此值的结果将被过滤。默认为0.0。
    - metric (str): 相似度度量方式: 'l2'(欧氏距离), 'cosine'(余弦相似度), 'ip'(内积)。默认为"cosine"。
    - api_key (Optional[str]): 用于嵌入API的密钥，如未提供则使用环境变量。默认为None。

    特殊使用限制或潜在的副作用:
    - 如果未提供`path`，则向量数据库将不会持久化保存。
    - `api_key`如果未提供，系统将尝试从环境变量中获取。
    """
    
    path: Optional[str] = Field(
        default=None, 
        description="向量数据库持久化的保存路径"
    )
    embedding_model: str = Field(
        default="BAAI/bge-m3", 
        description="用于文本嵌入的模型名称"
    )
    dimension: int = Field(
        default=512, 
        description="嵌入向量的维度"
    )
    top_k: int = Field(
        default=5, 
        description="查询时返回的最相似结果数量"
    )
    similarity_threshold: float = Field(
        default=0.0, 
        description="相似度阈值，低于此值的结果将被过滤"
    )
    metric: str = Field(
        default="cosine", 
        description="相似度度量方式: 'l2'(欧氏距离), 'cosine'(余弦相似度), 'ip'(内积)"
    )
    api_key: Optional[str] = Field(
        default=None, 
        description="用于嵌入API的密钥，如未提供则使用环境变量"
    )


class VectorFaissMemory(Memory):
    """
    基于FAISS向量数据库的记忆实现，用于存储和检索向量化的记忆内容。
    支持文本内容的语义搜索。

    核心功能包括：
    - 初始化向量数据库
    - 更新模型上下文，添加相关记忆内容
    - 查询记忆库并返回相关条目
    - 向记忆库添加新内容
    - 清除所有记忆条目
    - 清理记忆实现使用的资源

    使用示例：

    构造函数参数：
    - config: 配置参数，类型为VectorFaissMemoryConfig，包含数据库路径、向量维度、嵌入模型、度量标准、API密钥等配置信息。

    特殊使用限制或潜在的副作用：
    - 如果配置中设置了路径但数据库没有加载成功，则会创建目录以确保后续可以保存。
    - 在关闭前会自动保存数据库，如果保存失败会打印错误信息。
    """
    
    def __init__(self, config: VectorFaissMemoryConfig = VectorFaissMemoryConfig()):
        """
        初始化FAISS向量记忆
        
        Args:
            config: 配置参数
        """
        super().__init__()
        self.config = config
        
        # 初始化向量数据库
        self.db = VectorDB(
            path=config.path,
            dimension=config.dimension,
            embedding_model=config.embedding_model,
            metric=config.metric,
            api_key=config.api_key
        )
        
        # 如果设置了路径但数据库没有加载成功，则创建目录确保后续可以保存
        if config.path and len(self.db.texts) == 0:
            os.makedirs(os.path.dirname(os.path.abspath(config.path)), exist_ok=True)
    
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
                        source="VectorFaissMemory",
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
            MemoryQueryResult 包含记忆条目和相关性分数
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
        
        # 执行异步查询
        # 将同步搜索转换为异步操作
        search_results = await asyncio.to_thread(self.db.search, query_text, top_k)
        
        # 过滤结果，确保它们符合相似度阈值
        filtered_results = []
        for result in search_results:
            if result["score"] >= self.config.similarity_threshold:
                filtered_results.append(result)
        
        # 将搜索结果转换为MemoryContent
        memory_results = []
        for result in filtered_results:
            # 创建一个包含原始文本和元数据的记忆内容
            memory_content = MemoryContent(
                content=result["text"],
                mime_type=MemoryMimeType.TEXT,
                metadata={
                    "score": result["score"],
                    "id": result["id"],
                    "metric": result["metric"]
                }
            )
            memory_results.append(memory_content)
        
        return MemoryQueryResult(results=memory_results)
    
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
        
        # 提取文本内容
        text_content = ""
        if isinstance(content.content, str):
            text_content = content.content
        elif isinstance(content.content, dict) and "text" in content.content:
            text_content = content.content["text"]
        
        if not text_content:
            # 不处理空内容
            return
        
        # 将文本添加到向量数据库
        await asyncio.to_thread(self.db.add, text_content)
        
        # 如果设置了路径，自动保存数据库
        if self.config.path:
            await asyncio.to_thread(self.db.save, self.config.path)
    
    async def clear(self) -> None:
        """
        清除所有记忆条目
        """
        # 重新初始化数据库
        self.db = VectorDB(
            dimension=self.config.dimension,
            embedding_model=self.config.embedding_model,
            metric=self.config.metric,
            api_key=self.config.api_key
        )
        
        # 如果设置了路径，尝试删除旧文件
        if self.config.path:
            try:
                meta_path = f"{self.config.path}.meta"
                index_path = f"{self.config.path}.index"
                
                if os.path.exists(meta_path):
                    os.remove(meta_path)
                
                if os.path.exists(index_path):
                    os.remove(index_path)
            except Exception as e:
                print(f"清除记忆文件时出错: {e}")
    
    async def close(self) -> None:
        """
        清理记忆实现使用的资源
        """
        # 如果设置了路径，在关闭前保存数据库
        if self.config.path:
            try:
                await asyncio.to_thread(self.db.save, self.config.path)
            except Exception as e:
                print(f"关闭时保存数据库失败: {e}")
