import os
import json
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field

from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core._cancellation_token import CancellationToken
from autogen_agentchat.messages import TextMessage

class ListMemoryConfig(BaseModel):
    """
    列表存储记忆的配置类。
    
    该类用于配置列表存储记忆的相关参数，包括持久化路径和最大记忆条目数。
    列表记忆按照添加顺序存储记忆内容，主要用于保存历史对话或顺序操作记录。
    
    核心功能:
    - 配置列表记忆的持久化路径
    - 设置最大记忆条目数，超过此数量将移除最早的记忆
    - 设置记忆实例的名称标识
    
    构造函数参数:
    - name (Optional[str]): 记忆实例的名称。默认为 "default_list_memory"。
    - path (Optional[str]): 列表记忆持久化的保存路径。默认为 None。
    - max_items (int): 最大记忆条目数量，超过此数量将移除最早的记忆。默认为 100。
    
    特殊使用限制或潜在的副作用:
    - 如果 `path` 为 None，则记忆不会持久化保存。
    - `max_items` 应该大于 0，否则可能导致无法存储任何记忆。
    """
    
    name: Optional[str] = Field(
        default="default_list_memory", 
        description="记忆实例的名称"
    )
    path: Optional[str] = Field(
        default=None, 
        description="列表记忆持久化的保存路径"
    )
    max_items: int = Field(
        default=100, 
        description="最大记忆条目数量，超过此数量将移除最早的记忆"
    )


class ListMemory(Memory):
    """
    简单的基于列表的记忆实现，按时间顺序存储和检索记忆内容。
    
    该类将记忆内容存储在列表中，并按照时间顺序检索。通过update_context方法将记忆添加到模型上下文中。
    记忆内容可以通过content属性直接访问和修改，允许外部应用程序直接管理记忆内容。
    
    核心功能：
    - 按时间顺序存储和检索记忆内容
    - 更新模型上下文中的记忆内容
    - 直接访问和修改记忆列表
    - 支持从文件加载和保存记忆
    
    使用示例：
    ```python
    memory = ListMemory(name="聊天历史")
    await memory.add(MemoryContent(content="用户喜欢正式的语言", mime_type=MemoryMimeType.TEXT))
    
    # 直接修改记忆内容
    memory.content = [MemoryContent(content="新的偏好", mime_type=MemoryMimeType.TEXT)]
    
    # 更新模型上下文
    await memory.update_context(model_context)
    ```
    
    构造函数参数：
    - config: 配置参数，类型为ListMemoryConfig
    
    特殊使用限制或潜在的副作用：
    - 如果配置中设置了路径，会在初始化时尝试从文件加载记忆，并在关闭时保存记忆
    - 如果记忆数量超过最大限制，会自动移除最早的记忆
    """
    
    def __init__(self, config: ListMemoryConfig = ListMemoryConfig()):
        """
        初始化列表记忆
        
        Args:
            config: 配置参数
        """
        super().__init__()
        self.config = config
        self._name = config.name
        self._contents: List[MemoryContent] = []
        
        # 如果设置了路径，尝试从文件加载记忆
        if config.path and os.path.exists(config.path):
            self._load_from_file()
    
    @property
    def name(self) -> str:
        """
        获取记忆实例标识
        
        Returns:
            str: 记忆实例名称
        """
        return self._name
    
    @property
    def content(self) -> List[MemoryContent]:
        """
        获取当前记忆内容
        
        Returns:
            List[MemoryContent]: 存储的记忆内容列表
        """
        return self._contents
    
    @content.setter
    def content(self, value: List[MemoryContent]) -> None:
        """
        设置记忆内容
        
        Args:
            value: 要存储的新记忆内容列表
        """
        self._contents = value
        
        # 如果设置了路径，自动保存到文件
        if self.config.path:
            self._save_to_file()
    
    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """
        通过附加记忆内容更新模型上下文
        
        该方法将所有记忆作为新消息添加到提供的模型上下文末尾。
        
        Args:
            model_context: 要更新的上下文
            
        Returns:
            UpdateContextResult 包含添加到上下文中的记忆
        """
        if not self._contents:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))
        
        # 将记忆格式化为字符串列表
        memory_strings = []
        for i, memory in enumerate(self._contents, 1):
            if isinstance(memory.content, str):
                memory_strings.append(f"{i}. {memory.content}")
            elif isinstance(memory.content, dict) and "text" in memory.content:
                memory_strings.append(f"{i}. {memory.content['text']}")
            else:
                try:
                    memory_strings.append(f"{i}. {str(memory.content)}")
                except:
                    memory_strings.append(f"{i}. [无法显示的内容]")
        
        if memory_strings:
            memory_content = "历史记忆内容（按时间顺序）：\n" + "\n".join(memory_strings)
            
            # 将记忆内容添加到上下文最后
            model_context._messages.append(
                TextMessage(
                    source="ListMemory",
                    content=memory_content,
                    type="TextMessage"
                )
            )
        
        return UpdateContextResult(memories=MemoryQueryResult(results=self._contents))
    
    async def query(
        self,
        query: Union[str, MemoryContent],
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """
        返回所有记忆，不进行过滤
        
        Args:
            query: 在此实现中被忽略
            cancellation_token: 可选的取消操作令牌
            **kwargs: 额外的参数，如limit限制返回的记忆数量
            
        Returns:
            MemoryQueryResult 包含所有存储的记忆
        """
        if cancellation_token and cancellation_token._cancelled:
            return MemoryQueryResult(results=[])
        
        # 可选的限制返回数量
        limit = kwargs.get("limit", len(self._contents))
        
        # 返回最新的n条记忆
        if limit < len(self._contents):
            return MemoryQueryResult(results=self._contents[-limit:])
        
        return MemoryQueryResult(results=self._contents)
    
    async def add(self, content: MemoryContent, cancellation_token: Optional[CancellationToken] = None) -> None:
        """
        向记忆中添加新内容
        
        Args:
            content: 要存储的记忆内容
            cancellation_token: 可选的取消操作令牌
        """
        if cancellation_token and cancellation_token._cancelled:
            return
        
        # 添加到列表
        self._contents.append(content)
        
        # 如果超过最大记忆数量，移除最早的记忆
        while len(self._contents) > self.config.max_items:
            self._contents.pop(0)
        
        # 如果设置了路径，保存到文件
        if self.config.path:
            self._save_to_file()
    
    async def clear(self) -> None:
        """清除所有记忆内容"""
        self._contents = []
        
        # 如果设置了路径，删除文件
        if self.config.path and os.path.exists(self.config.path):
            try:
                os.remove(self.config.path)
            except Exception as e:
                print(f"清除记忆文件时出错: {e}")
    
    async def close(self) -> None:
        """清理资源（如果需要）"""
        # 如果设置了路径，确保在关闭前保存数据
        if self.config.path:
            self._save_to_file()
    
    def _save_to_file(self) -> None:
        """将记忆保存到文件"""
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(os.path.abspath(self.config.path)), exist_ok=True)
            
            # 准备要保存的数据
            serializable_memories = []
            for memory in self._contents:
                # 序列化MemoryContent
                memory_dict = {
                    "content": memory.content if isinstance(memory.content, (str, bytes)) else json.dumps(memory.content),
                    "mime_type": memory.mime_type.value if isinstance(memory.mime_type, MemoryMimeType) else memory.mime_type,
                    "metadata": memory.metadata
                }
                serializable_memories.append(memory_dict)
            
            # 以JSON格式保存
            with open(self.config.path, 'w', encoding='utf-8') as f:
                json.dump(serializable_memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆到文件时出错: {e}")
    
    def _load_from_file(self) -> None:
        """从文件加载记忆"""
        try:
            with open(self.config.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载记忆内容
            self._contents = []
            for memory_dict in data:
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
                self._contents.append(memory)
        except Exception as e:
            print(f"从文件加载记忆时出错: {e}")
            # 初始化空记忆
            self._contents = []
