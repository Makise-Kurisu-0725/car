import os
from typing import Optional

from pydantic import BaseModel, Field
from autogen_core.memory import Memory

from .list_memory import ListMemory, ListMemoryConfig
from .dict_memory import DictMemory, DictMemoryConfig
from .vector_faiss_memory import VectorFaissMemory, VectorFaissMemoryConfig
from .utils import LLMClient


class BaseAgentMemoryConfig(BaseModel):
    """
    智能体基础记忆配置类
    
    该类整合了多种记忆类型的配置，为智能体提供完整的记忆管理功能。
    每种记忆类型都有专门的配置部分，可以根据需要启用或禁用。
    
    核心功能：
    - 配置用户画像记忆
    - 配置短期记忆
    - 配置对话原始数据记忆
    - 配置任务总结/反思记忆
    - 配置用于总结反思的大语言模型API客户端
    
    使用示例：
    
    构造函数参数：
    - name: 记忆实例的名称
    - base_path: 记忆持久化的基础路径，各子记忆将基于此路径创建自己的文件
    - enable_user_profile: 是否启用用户画像记忆
    - user_profile_config: 用户画像记忆配置
    - enable_short_term_memory: 是否启用短期记忆
    - short_term_memory_config: 短期记忆配置
    - enable_conversation_memory: 是否启用对话原始数据记忆
    - conversation_memory_config: 对话原始数据记忆配置
    - enable_reflection_memory: 是否启用任务总结/反思记忆
    - reflection_memory_config: 任务总结/反思记忆配置
    - use_memory: 在响应时是否使用记忆
    - llm_api_key: 大语言模型API密钥，如未提供则使用环境变量
    - llm_base_url: 大语言模型API基础URL
    - llm_model: 使用的大语言模型名称
    
    特殊使用限制或潜在的副作用：
    - 如果未提供llm_api_key，则必须确保环境变量API_KEY已正确设置。
    - base_path必须是一个有效的文件路径，否则可能导致记忆持久化失败。
    """
    
    # 基础配置
    name: str = Field(
        default="agent_memory",
        description="记忆实例的名称"
    )
    base_path: Optional[str] = Field(
        default=None,
        description="记忆持久化的基础路径，各子记忆将基于此路径创建自己的文件"
    )
    
    # 用户特征/画像配置
    enable_user_profile: bool = Field(
        default=True,
        description="是否启用用户画像记忆"
    )
    user_profile_config: ListMemoryConfig = Field(
        default_factory=lambda: ListMemoryConfig(
            name="user_profile",
            max_items=100
        ),
        description="用户画像记忆配置"
    )
    
    # 短期记忆配置
    enable_short_term_memory: bool = Field(
        default=True,
        description="是否启用短期记忆"
    )
    short_term_memory_config: DictMemoryConfig = Field(
        default_factory=lambda: DictMemoryConfig(
            query_match_method="keyword",
            max_items=50
        ),
        description="短期记忆配置"
    )
    
    # 对话原始数据配置
    enable_conversation_memory: bool = Field(
        default=True,
        description="是否启用对话原始数据记忆"
    )
    conversation_memory_config: VectorFaissMemoryConfig = Field(
        default_factory=lambda: VectorFaissMemoryConfig(
            embedding_model="BAAI/bge-m3",
            dimension=1024,
            top_k=2
        ),
        description="对话原始数据记忆配置"
    )
    
    # 任务总结/反思配置
    enable_reflection_memory: bool = Field(
        default=True,
        description="是否启用任务总结/反思记忆"
    )
    reflection_memory_config: VectorFaissMemoryConfig = Field(
        default_factory=lambda: VectorFaissMemoryConfig(
            embedding_model="BAAI/bge-m3",
            dimension=1024,
            top_k=2,
            similarity_threshold=0.5
        ),
        description="任务总结/反思记忆配置"
    )

    use_memory: bool = Field(
        default=True,
        description="在响应时是否使用记忆"
    )
    
    # 用于生成反思的大语言模型配置
    # LLM客户端配置
    llm_api_key: Optional[str] = Field(
        default=os.getenv("API_KEY"),
        description="大语言模型API密钥，如未提供则使用环境变量"
    )
    llm_base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        description="大语言模型API基础URL"
    )
    llm_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="使用的大语言模型名称"
    )



class BaseAgentMemory(Memory):
    """
    智能体基础记忆类
    
    整合多种记忆类型，为智能体提供完整的记忆管理功能。包含以下记忆组件：
    1. 用户特征/画像 (ListMemory): 存储用户长期的特征、偏好和画像信息
    2. 短期记忆 (DictMemory): 存储近期的对话历史和上下文信息
    3. 对话原始数据 (VectorFaissMemory): 存储所有对话的原始内容，支持语义搜索
    4. 任务总结/反思 (VectorFaissMemory): 存储对任务完成情况的总结和反思
    
    每个组件可以单独访问和管理，同时BaseAgentMemory类也提供了统一的接口进行查询和操作。
    
    设计灵活，支持子类扩展添加其他记忆组件。
    
    Args:
        config (BaseAgentMemoryConfig): 记忆配置，默认为BaseAgentMemoryConfig的实例。
        
    使用示例：
    """
    
    def __init__(self, config: BaseAgentMemoryConfig = BaseAgentMemoryConfig()):
        """
        初始化智能体基础记忆
        
        Args:
            config: 记忆配置
        """
        super().__init__()
        self.config = config
        
        # 设置持久化路径
        self._setup_paths()
        
        # 初始化各组件
        self._init_components()
        
        # 初始化LLM客户端
        self.llm_client = LLMClient(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            model=config.llm_model
        )
        
        # 默认使用内置的异步反思生成器
        self._reflection_generator = self._default_reflection_generator
    
    def _setup_paths(self) -> None:
        """设置各组件的持久化路径"""
        if self.config.base_path:
            base_path = self.config.base_path
            
            # 确保基础目录存在
            os.makedirs(base_path, exist_ok=True)
            
            # 用户画像路径
            if self.config.enable_user_profile:
                self.config.user_profile_config.path = os.path.join(base_path, "user_profile.json")
                
            # 短期记忆路径
            if self.config.enable_short_term_memory:
                self.config.short_term_memory_config.path = os.path.join(base_path, "short_term_memory.json")
                
            # 对话原始数据路径
            if self.config.enable_conversation_memory:
                self.config.conversation_memory_config.path = os.path.join(base_path, "conversation_memory")
                
            # 任务反思路径
            if self.config.enable_reflection_memory:
                self.config.reflection_memory_config.path = os.path.join(base_path, "reflection_memory")
    
    def _init_components(self) -> None:
        """初始化记忆组件"""
        # 用户特征/画像
        self.user_profile = ListMemory(self.config.user_profile_config) if self.config.enable_user_profile else None
        
        # 短期记忆
        self.short_term_memory = DictMemory(self.config.short_term_memory_config) if self.config.enable_short_term_memory else None
        
        # 对话原始数据
        self.conversation_memory = VectorFaissMemory(self.config.conversation_memory_config) if self.config.enable_conversation_memory else None
        
        # 任务总结/反思
        self.reflection_memory = VectorFaissMemory(self.config.reflection_memory_config) if self.config.enable_reflection_memory else None
        
        # 存储所有启用的记忆组件
        self.memory_components = {}
        if self.user_profile:
            self.memory_components['user_profile'] = self.user_profile
        if self.short_term_memory:
            self.memory_components['short_term_memory'] = self.short_term_memory
        if self.conversation_memory:
            self.memory_components['conversation_memory'] = self.conversation_memory
        if self.reflection_memory:
            self.memory_components['reflection_memory'] = self.reflection_memory

# 内存适配器类，包装BaseAgentMemory以提供兼容接口
class MemoryAdapter:
    """
    内存适配器类
    
    包装BaseAgentMemory实例，提供兼容不同Memory接口的适配功能。
    用于解决类型兼容性问题。
    
    Attributes:
        agent_memory (BaseAgentMemory): 被包装的内存实例，提供基本的内存操作。
    
    Methods:
        update_context(model_context): 更新模型上下文。
        query(query, cancellation_token=None, **kwargs): 查询内存中的信息。
        add(content, cancellation_token=None): 向内存中添加内容。
        clear(): 清空内存中的所有内容。
        close(): 关闭内存连接或释放资源。
    
    Example:
    
    Args:
        agent_memory (BaseAgentMemory): 被包装的内存实例，必须实现update_context, query, add, clear, close等方法。
    
    Note:
        - 构造函数的参数`agent_memory`必须是一个实现了相应方法的实例。
        - 该类主要用于解决不同内存接口的兼容性问题，不提供额外的功能。
    """
    def __init__(self, agent_memory):
        self.agent_memory = agent_memory
    
    async def update_context(self, model_context):
        return await self.agent_memory.update_context(model_context)
    
    async def query(self, query, cancellation_token=None, **kwargs):
        return await self.agent_memory.query(query, cancellation_token, **kwargs)
    
    async def add(self, content, cancellation_token=None):
        return await self.agent_memory.add(content, cancellation_token)
    
    async def clear(self):
        return await self.agent_memory.clear()
    
    async def close(self):
        return await self.agent_memory.close()
