import os
import json
import asyncio
from typing import Sequence, List, AsyncGenerator, Dict, Any, Optional, Union

from autogen_core import CancellationToken
from autogen_agentchat.base import Response
from autogen_agentchat.messages import AgentEvent, ChatMessage

from .baseagent import BaseAgent
from memory_module.base_agent_memory import BaseAgentMemory, BaseAgentMemoryConfig


class BaseMemoryAgent(BaseAgent):
    """
    具有增强记忆能力的基础智能体
    
    扩展BaseAgent并使用BaseAgentMemory进行记忆管理，提供更强大的记忆功能。
    整合了用户画像、短期记忆、对话记忆和任务反思等多种记忆类型，
    使智能体能够根据历史交互更好地理解和响应用户请求。
    
    核心功能：
    - 初始化时创建并配置记忆系统
    - 处理接收到的消息时更新记忆
    - 流式处理接收到的消息时更新记忆
    - 任务成功后生成反思并添加到记忆
    - 从消息历史中提取用户特征
    - 确保资源释放
    
    使用示例：
    
    构造函数参数：
    - name: 智能体名称
    - description: 智能体描述
    - device_type: 设备类型
    - memory_config: 记忆配置，如未提供则使用默认配置
    - memory_base_path: 记忆持久化的基础路径
    - check_mode: 是否启用结果检查模式
    - max_retries: 最大重试次数
    - max_tool_retries: 工具调用最大重试次数
    - device_manager: 设备管理器，如未提供则使用默认管理器
    - **kwargs: 传递给BaseAgent的额外参数
    
    特殊使用限制或潜在的副作用：
    - 在初始化时，如果未提供memory_config，将自动创建一个默认配置。
    - 如果提供了memory_base_path，记忆将持久化到指定路径。
    - 在处理消息时，如果use_memory为True，将更新记忆系统。
    - 在任务成功后，将异步生成反思并添加到记忆。
    - 从消息历史中提取用户特征时，需要提供消息历史列表。
    - 确保在资源使用完毕后调用__aexit__方法释放资源。
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        device_type: str,
        memory_config: Optional[BaseAgentMemoryConfig] = None,
        memory_base_path: Optional[str] = None,
        check_mode: bool = True,
        max_retries: int = 3,
        max_tool_retries: int = 2,
        device_manager=None,  
        **kwargs
    ):
        """
        初始化具有增强记忆能力的基础智能体
        
        Args:
            name: 智能体名称
            description: 智能体描述
            device_type: 设备类型
            memory_config: 记忆配置，如未提供则使用默认配置
            memory_base_path: 记忆持久化的基础路径
            check_mode: 是否启用结果检查模式
            max_retries: 最大重试次数
            max_tool_retries: 工具调用最大重试次数
            device_manager: 设备管理器，如未提供则使用默认管理器
            **kwargs: 传递给BaseAgent的额外参数
        """
        # 创建记忆配置
        if memory_config is None:
            memory_config = BaseAgentMemoryConfig(name=f"{name}_memory")
        
        # 设置记忆持久化路径
        if memory_base_path is not None:
            memory_config.base_path = os.path.join(memory_base_path, name)
        
        # 初始化记忆系统（但不作为参数传递给父类）
        self.agent_memory = BaseAgentMemory(config=memory_config)
        
        # 移除memory参数，如果存在的话
        if 'memory' in kwargs:
            del kwargs['memory']
        
        # 设置device_manager属性，避免BaseAgent初始化时出错
        self.device_type = device_type
        self.device_manager = device_manager  # 确保在调用super().__init__之前设置
        
        # 初始化基类（不传递memory参数）
        super().__init__(
            name=name,
            description=description,
            device_type=device_type,
            check_mode=check_mode,
            max_retries=max_retries,
            max_tool_retries=max_tool_retries,
            **kwargs
        )
        
        # 手动设置_memory属性，跳过类型检查
        # 注意：这是一个内部变量，通常不应直接访问，但我们需要绕过类型检查
        self._memory = [self.agent_memory]
        self.use_memory = memory_config.use_memory
    
    async def on_messages(
        self, 
        messages: Sequence[ChatMessage], 
        cancellation_token: CancellationToken,
    ) -> Response:
        """
        处理接收到的消息
        
        重写基类方法，增加记忆处理逻辑
        
        Args:
            messages: 接收到的消息序列
            cancellation_token: 取消操作令牌
            use_memory: 是否使用记忆来增强响应，默认为True
            
        Returns:
            智能体响应
        """
        # 处理新消息，更新记忆
        if self.use_memory:
            for message in messages:
                if hasattr(message, 'content') and message.content:
                    # 将消息转换为字典格式
                    message_dict = {
                        'role': message.source if hasattr(message, 'source') else 'user',
                        'content': message.content
                    }
                    # 更新记忆
                    await self.agent_memory.process_new_message(message_dict, cancellation_token)
        
        # 调用基类方法处理消息
        # 注意：基类的on_messages方法会调用_update_model_context_with_memory，
        # 该方法依赖于传递给BaseAgent构造函数的memory参数(即self.agent_memory)
        response = await super().on_messages(messages, cancellation_token)
        
        # 如果响应成功，可以生成任务反思
        if (self.use_memory and 
            hasattr(response.chat_message, 'metadata') and 
            response.chat_message.metadata.get('label') == 'success' and
            len(messages) > 0):
            # 异步生成反思（不阻塞响应）
            asyncio.create_task(self._generate_reflection_after_success(
                query=messages[0].content if hasattr(messages[0], 'content') else "",
                response=response,
                cancellation_token=cancellation_token
            ))
        
        return response
    
    async def on_messages_stream(
        self, 
        messages: Sequence[ChatMessage], 
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:
        """
        流式处理接收到的消息
        
        重写基类方法，增加记忆处理逻辑
        
        Args:
            messages: 接收到的消息序列
            cancellation_token: 取消操作令牌
            use_memory: 是否使用记忆来增强响应，默认为True
            
        Yields:
            代理事件、聊天消息或响应
        """
        # 处理新消息，更新记忆
        if self.use_memory:
            for message in messages:
                if hasattr(message, 'content') and message.content:
                    # 将消息转换为字典格式
                    message_dict = {
                        'role': message.source if hasattr(message, 'source') else 'user',
                        'content': message.content
                    }
                    # 更新记忆
                    await self.agent_memory.process_new_message(message_dict, cancellation_token)
        
        # 使用父类的流式处理方法
        async for event in super().on_messages_stream(messages, cancellation_token):
            yield event
            
            # 如果是最终响应并且成功，异步生成反思
            if (self.use_memory and 
                isinstance(event, Response) and 
                hasattr(event.chat_message, 'metadata') and 
                event.chat_message.metadata.get('label') == 'success' and
                len(messages) > 0):
                # 异步生成反思（不阻塞响应）
                asyncio.create_task(self._generate_reflection_after_success(
                    query=messages[0].content if hasattr(messages[0], 'content') else "",
                    response=event,
                    cancellation_token=cancellation_token
                ))
    
    async def _generate_reflection_after_success(self, query: str, response: Response, cancellation_token: CancellationToken) -> None:
        """
        任务成功后生成反思并添加到记忆
        
        Args:
            query: 用户查询
            response: 智能体响应
            cancellation_token: 取消操作令牌
        """
        if cancellation_token and cancellation_token._cancelled:
            return
        
        # 构建消息历史
        message_history = [
            {'role': 'user', 'content': query},
            {'role': 'assistant', 'content': str(response.chat_message)}
        ]
        
        # 生成反思
        await self.agent_memory.generate_reflection(
            task_description=query,
            message_history=message_history,
            cancellation_token=cancellation_token
        )
    
    async def extract_user_profile_from_history(
        self, 
        message_history: List[Dict[str, Any]], 
        cancellation_token: Optional[CancellationToken] = None
    ) -> None:
        """
        从消息历史中提取用户特征
        
        Args:
            message_history: 历史消息列表
            cancellation_token: 取消操作令牌
        """
        if cancellation_token and cancellation_token._cancelled:
            return
        
        # 调用记忆模块的用户特征提取方法
        await self.agent_memory.extract_user_profile(message_history, cancellation_token)
    
    async def __aexit__(self, *args) -> None:
        """
        确保资源释放
        """
        # 关闭记忆系统
        await self.agent_memory.close()
        # 调用基类方法
        await super().__aexit__(*args)
