本记忆系统实现提供了一系列灵活、可扩展的记忆组件，用于增强智能体的持久化记忆能力。模块支持多种不同类型的记忆存储和检索方式，可以根据不同的应用场景进行组合和定制。

## 文件夹组织结构

```
memory_module/
├── base_agent_memory.py  # 基础智能体记忆类和配置
├── list_memory.py        # 基于列表的记忆实现
├── dict_memory.py        # 基于字典的记忆实现
├── vector_faiss_memory.py # 基于FAISS的向量记忆实现
└── utils.py              # 工具类，包含LLM客户端、嵌入模型API和向量数据库
```

## 文件内容介绍

### base_agent_memory.py

提供基础智能体记忆框架，整合多种记忆类型为智能体提供完整记忆管理功能。主要包含：

- `BaseAgentMemoryConfig`: 智能体基础记忆配置类，整合多种记忆类型的配置
- `BaseAgentMemory`: 智能体基础记忆类，管理多种记忆组件
- `MemoryAdapter`: 记忆适配器类，提供兼容不同Memory接口的适配功能

### list_memory.py

实现基于列表的简单记忆存储，适合存储按时间顺序的记忆内容。主要包含：

- `ListMemoryConfig`: 列表记忆配置类
- `ListMemory`: 基于列表的记忆实现，按添加顺序存储和检索记忆内容

### dict_memory.py

实现基于字典的记忆存储，支持关键词和子字符串搜索。主要包含：

- `DictMemoryConfig`: 字典记忆配置类
- `DictMemory`: 基于字典的记忆实现，提供基于关键词和子字符串的检索能力

### vector_faiss_memory.py

实现基于FAISS的向量记忆存储，支持语义相似度搜索。主要包含：

- `VectorFaissMemoryConfig`: FAISS向量记忆配置类
- `VectorFaissMemory`: 基于FAISS的向量记忆实现，提供语义搜索能力

### utils.py

提供各种工具类，支持记忆模块的功能实现。主要包含：

- `LLMClient`: 大语言模型API客户端，提供模型调用和反思生成功能
- `EmbeddingModelAPI`: 嵌入模型API，用于将文本转换为向量
- `VectorDB`: 基于FAISS的向量数据库类，支持文本的存储、检索、删除和持久化

## 各记忆类型使用方法

### 列表记忆 (ListMemory)

适用场景：存储用户特征、偏好、画像等按时间顺序的信息。

```python
from memory_module.list_memory import ListMemory, ListMemoryConfig

# 基本配置
config = ListMemoryConfig(
    name="用户画像",  # 记忆实例名称
    path="./memory/user_profile.json",  # 持久化路径
    max_items=100  # 最大记忆条目数
)

# 创建记忆实例
memory = ListMemory(config)

# 添加记忆
from autogen_core.memory import MemoryContent, MemoryMimeType
await memory.add(MemoryContent(content="用户喜欢正式的语言风格", mime_type=MemoryMimeType.TEXT))

# 查询所有记忆
memory_results = await memory.query("查询无关紧要，返回所有内容")
```

### 字典记忆 (DictMemory)

适用场景：存储短期记忆、对话上下文等需要基于关键词或子字符串查询的信息。

```python
from memory_module.dict_memory import DictMemory, DictMemoryConfig

# 基本配置
config = DictMemoryConfig(
    path="./memory/short_term.json",  # 持久化路径
    max_items=50,  # 最大记忆条目数
    query_match_method="keyword",  # 查询匹配方法: 'substring'(子字符串), 'exact'(精确), 'keyword'(关键词)
    top_k=5  # 返回结果数量
)

# 创建记忆实例
memory = DictMemory(config)

# 添加记忆
from autogen_core.memory import MemoryContent, MemoryMimeType
await memory.add(MemoryContent(content="用户刚刚询问了关于Python的问题", mime_type=MemoryMimeType.TEXT))

# 基于关键词查询记忆
memory_results = await memory.query("Python问题")
```

### 向量记忆 (VectorFaissMemory)

适用场景：存储对话原始数据、任务总结/反思等需要语义搜索的信息。

```python
from memory_module.vector_faiss_memory import VectorFaissMemory, VectorFaissMemoryConfig

# 基本配置
config = VectorFaissMemoryConfig(
    path="./memory/conversation",  # 持久化路径
    embedding_model="BAAI/bge-m3",  # 嵌入模型
    dimension=1024,  # 向量维度
    top_k=5,  # 返回结果数量
    similarity_threshold=0.5,  # 相似度阈值
    metric="cosine"  # 相似度度量方式: 'l2'(欧氏距离), 'cosine'(余弦相似度), 'ip'(内积)
)

# 创建记忆实例
memory = VectorFaissMemory(config)

# 添加记忆
from autogen_core.memory import MemoryContent, MemoryMimeType
await memory.add(MemoryContent(content="用户询问如何在Python中使用多线程", mime_type=MemoryMimeType.TEXT))

# 语义查询记忆
memory_results = await memory.query("Python并发编程")
```

### 整合多种记忆 (BaseAgentMemory)

```python
from memory_module.base_agent_memory import BaseAgentMemory, BaseAgentMemoryConfig

# 创建配置
config = BaseAgentMemoryConfig(
    name="agent_memory",
    base_path="./memory",  # 基础路径
    
    # 用户画像记忆配置
    enable_user_profile=True,
    user_profile_config=ListMemoryConfig(name="user_profile", max_items=100),
    
    # 短期记忆配置
    enable_short_term_memory=True,
    short_term_memory_config=DictMemoryConfig(query_match_method="keyword", max_items=50),
    
    # 对话原始数据配置
    enable_conversation_memory=True,
    conversation_memory_config=VectorFaissMemoryConfig(
        embedding_model="BAAI/bge-m3", 
        dimension=1024, 
        top_k=2
    ),
    
    # 任务反思配置
    enable_reflection_memory=True,
    reflection_memory_config=VectorFaissMemoryConfig(
        embedding_model="BAAI/bge-m3", 
        dimension=1024, 
        top_k=2,
        similarity_threshold=0.5
    )
)

# 创建记忆实例
memory = BaseAgentMemory(config)

# 直接访问各子记忆
memory.user_profile.content  # 访问用户画像
await memory.short_term_memory.add(...)  # 添加短期记忆
await memory.conversation_memory.query(...)  # 查询对话记忆
```

## 基于BaseAgentMemory定制化自己的记忆

要定制自己的记忆系统，可以继承`BaseAgentMemory`类并按需扩展功能。以下是定制化的基本步骤：

### 1. 创建自定义记忆配置类

```python
from pydantic import Field
from memory_module.base_agent_memory import BaseAgentMemoryConfig

class CustomAgentMemoryConfig(BaseAgentMemoryConfig):
    # 添加自定义配置
    enable_custom_memory: bool = Field(
        default=True,
        description="是否启用自定义记忆"
    )
    custom_memory_config: YourCustomMemoryConfig = Field(
        default_factory=lambda: YourCustomMemoryConfig(name="custom_memory"),
        description="自定义记忆配置"
    )
```

### 2. 创建自定义记忆类

```python
from memory_module.base_agent_memory import BaseAgentMemory

class CustomAgentMemory(BaseAgentMemory):
    def __init__(self, config: CustomAgentMemoryConfig = CustomAgentMemoryConfig()):
        # 调用父类初始化
        super().__init__(config)
        self.config = config
        
        # 初始化自定义记忆组件
        if config.enable_custom_memory:
            self.custom_memory = YourCustomMemory(config.custom_memory_config)
            self.memory_components['custom_memory'] = self.custom_memory
    
    # 重写或扩展方法
    async def custom_method(self, query):
        # 实现自定义功能
        pass
        
    # 可以重写默认反思生成器
    async def _default_reflection_generator(self, task_description, message_history):
        # 自定义反思生成逻辑
        return "自定义反思内容"
```

### 3. 实现自定义记忆组件

如果需要全新的记忆类型，可以实现自己的记忆组件：

```python
from autogen_core.memory import Memory, MemoryContent, MemoryQueryResult, UpdateContextResult
from pydantic import BaseModel, Field

# 定义配置类
class YourCustomMemoryConfig(BaseModel):
    name: str = Field(default="custom_memory", description="自定义记忆名称")
    # 添加其他配置参数...

# 实现记忆类
class YourCustomMemory(Memory):
    def __init__(self, config: YourCustomMemoryConfig):
        super().__init__()
        self.config = config
        # 初始化所需资源...
    
    async def update_context(self, model_context):
        # 实现上下文更新逻辑
        pass
    
    async def query(self, query, cancellation_token=None, **kwargs):
        # 实现查询逻辑
        pass
    
    async def add(self, content, cancellation_token=None):
        # 实现添加逻辑
        pass
    
    async def clear(self):
        # 实现清除逻辑
        pass
    
    async def close(self):
        # 实现关闭逻辑，释放资源
        pass
```

### 4. 使用自定义记忆

```python
# 创建自定义配置
config = CustomAgentMemoryConfig(
    name="my_agent_memory",
    base_path="./my_memory",
    # 设置其他配置项...
)

# 创建自定义记忆实例
memory = CustomAgentMemory(config)

# 使用自定义方法
await memory.custom_method("查询内容")
```

按以上步骤，可以根据具体需求定制化自己的边缘智能体记忆系统，添加新的记忆类型或扩展现有功能。
