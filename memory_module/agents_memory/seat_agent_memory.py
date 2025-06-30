import re
from typing import Dict, List, Optional, Any

from pydantic import Field, BaseModel

from autogen_core.memory import MemoryContent, MemoryMimeType, CancellationToken
from ..base_agent_memory import BaseAgentMemory, BaseAgentMemoryConfig


class SeatAgentMemoryConfig(BaseAgentMemoryConfig):
    """
    座椅智能体记忆配置类

    扩展基础记忆配置，增加座椅智能体特有的记忆管理配置。
    该类用于配置座椅智能体的记忆管理，包括座椅偏好记忆、座椅使用历史记忆和场景模式记忆。
    通过启用或禁用这些记忆配置，可以控制座椅智能体如何记录和管理用户的使用习惯和偏好设置。

    核心功能:
    - enable_seat_preference_memory: 是否启用座椅偏好记忆。
    - enable_seat_usage_history: 是否启用座椅使用历史记忆。
    - enable_scene_mode_memory: 是否启用场景模式记忆，记录不同场景下的座椅设置组合。

    使用示例:

    构造函数参数:
    - enable_seat_preference_memory (bool): 是否启用座椅偏好记忆，默认为True。
    - enable_seat_usage_history (bool): 是否启用座椅使用历史记忆，默认为True。
    - enable_scene_mode_memory (bool): 是否启用场景模式记忆，记录不同场景下的座椅设置组合，默认为True。

    特殊使用限制或潜在的副作用:
    - 禁用某些记忆配置可能会导致座椅智能体无法提供个性化的使用体验。
    - 启用所有记忆配置可能会增加内存使用和数据处理负担。
    """
    
    # 座椅偏好记忆配置
    enable_seat_preference_memory: bool = Field(
        default=True,
        description="是否启用座椅偏好记忆"
    )
    
    # 座椅使用历史记忆配置
    enable_seat_usage_history: bool = Field(
        default=True,
        description="是否启用座椅使用历史记忆"
    )
    
    # 场景模式记忆配置
    enable_scene_mode_memory: bool = Field(
        default=True,
        description="是否启用场景模式记忆，记录不同场景下的座椅设置组合"
    )


class SeatPreference(BaseModel):
    """
    座椅偏好数据模型

    该类用于表示用户的座椅偏好设置，包括座椅位置、按摩强度、通风强度和按摩模式等。

    核心功能:
    - 存储用户的座椅偏好设置
    - 允许设置和获取座椅的各种偏好参数

    使用示例:

    构造函数参数:
    - user_id (str): 用户ID
    - seat_position (str): 座椅位置，例如 "driver" 或 "passenger"
    - massage_intensity (Optional[int]): 按摩强度，范围从 0 到 5
    - ventilation_intensity (Optional[int]): 通风强度，范围从 0 到 5
    - massage_mode (Optional[str]): 按摩模式，例如 "relax" 或 "active"
    - last_updated (str): 最后更新时间，ISO 8601 格式字符串

    特殊使用限制或潜在的副作用:
    - 所有参数在创建实例时必须提供，除了 `massage_intensity`, `ventilation_intensity` 和 `massage_mode` 可以选填
    - `last_updated` 参数必须是有效的 ISO 8601 格式时间字符串
    """
    user_id: str
    seat_position: str
    massage_intensity: Optional[int] = None
    ventilation_intensity: Optional[int] = None
    massage_mode: Optional[str] = None
    last_updated: str


class SeatAgentMemory(BaseAgentMemory):
    """
    座椅智能体记忆类
    
    扩展智能体基础记忆，专门处理和存储座椅智能体相关的记忆信息。
    包括用户座椅偏好、座椅使用历史、常用场景模式等。
    
    Attributes:
        seat_positions (dict): 座椅位置映射，将位置标识映射到中文名称。
        config (SeatAgentMemoryConfig): 座椅智能体记忆配置对象。
        seat_preference (DictMemory): 存储用户对不同座位的偏好设置。
        seat_usage_history (VectorFaissMemory): 跟踪座椅的历史调整记录。
        scene_mode_memory (VectorFaissMemory): 存储不同场景下的座椅设置组合。
    
    Methods:
        __init__(self, config: SeatAgentMemoryConfig = SeatAgentMemoryConfig()): 初始化座椅智能体记忆。
        _init_seat_components(self) -> None: 初始化座椅特有记忆组件。
        process_new_message(self, message: Dict[str, Any], cancellation_token: Optional[CancellationToken] = None) -> None: 处理新消息，提取座椅相关命令和状态。
        _extract_seat_command(self, text: str) -> Optional[Dict[str, Any]]: 从文本中提取座椅命令。
        extract_user_profile(self, message_history: List[Dict], cancellation_token: Optional[CancellationToken] = None) -> None: 从消息历史中提取用户座椅偏好。
        add_to_seat_usage_history(self, seat_position: str, command_type: str, value: Any, user_id: str = "unknown", cancellation_token: Optional[CancellationToken] = None) -> None: 添加座椅使用记录到历史。
        save_scene_mode(self, scene_name: str, seat_settings: Dict[str, Dict[str, Any]], description: str = "", cancellation_token: Optional[CancellationToken] = None) -> None: 保存场景模式。
        get_user_seat_preference(self, user_id: str, seat_position: str, cancellation_token: Optional[CancellationToken] = None) -> Optional[Dict[str, Any]]: 获取用户对特定座椅的偏好设置。
        _get_current_time(self) -> float: 获取当前时间戳。
    
    Example:
    
    Args:
        config: 座椅智能体记忆配置。
    
    Note:
        - 该类依赖于外部配置对象 `SeatAgentMemoryConfig`。
        - 使用前需要确保相关依赖库已正确安装和配置。
    """
    
    def __init__(self, config: SeatAgentMemoryConfig = SeatAgentMemoryConfig()):
        """
        初始化座椅智能体记忆
        
        Args:
            config: 座椅智能体记忆配置
        """
        super().__init__(config)
        self.config = config
        
        # 座椅位置映射
        self.seat_positions = {
            "driver": "驾驶座",
            "passenger": "副驾驶座",
            "rear_left": "左后座",
            "rear_right": "右后座"
        }
        
        # 初始化座椅特有记忆组件
        self._init_seat_components()
    
    def _init_seat_components(self) -> None:
        """初始化座椅特有记忆组件"""
        # 座椅偏好记忆 - 存储用户对不同座位的偏好设置
        if self.config.enable_seat_preference_memory:
            from ..dict_memory import DictMemory, DictMemoryConfig
            
            seat_preference_config = DictMemoryConfig(
                name="seat_preference",
                query_match_method="keyword",
                max_items=100
            )
            
            if self.config.base_path:
                seat_preference_config.path = f"{self.config.base_path}/seat_preference.json"
                
            self.seat_preference = DictMemory(seat_preference_config)
            self.register_memory_component("seat_preference", self.seat_preference)
        
        # 座椅使用历史记忆 - 跟踪座椅的历史调整记录
        if self.config.enable_seat_usage_history:
            from ..vector_faiss_memory import VectorFaissMemory, VectorFaissMemoryConfig
            
            seat_usage_config = VectorFaissMemoryConfig(
                name="seat_usage_history",
                embedding_model="BAAI/bge-m3",
                dimension=1024,
                top_k=5
            )
            
            if self.config.base_path:
                seat_usage_config.path = f"{self.config.base_path}/seat_usage_history"
                
            self.seat_usage_history = VectorFaissMemory(seat_usage_config)
            self.register_memory_component("seat_usage_history", self.seat_usage_history)
        
        # 场景模式记忆 - 存储不同场景下的座椅设置组合
        if self.config.enable_scene_mode_memory:
            from ..vector_faiss_memory import VectorFaissMemory, VectorFaissMemoryConfig
            
            scene_mode_config = VectorFaissMemoryConfig(
                name="scene_mode_memory",
                embedding_model="BAAI/bge-m3",
                dimension=1024,
                top_k=3,
                similarity_threshold=0.65
            )
            
            if self.config.base_path:
                scene_mode_config.path = f"{self.config.base_path}/scene_mode_memory"
                
            self.scene_mode_memory = VectorFaissMemory(scene_mode_config)
            self.register_memory_component("scene_mode_memory", self.scene_mode_memory)
    
    async def process_new_message(self, message: Dict[str, Any], cancellation_token: Optional[CancellationToken] = None) -> None:
        """
        处理新消息，提取座椅相关命令和状态
        
        Args:
            message: 消息内容
            cancellation_token: 可选的取消操作令牌
        """
        # 先调用父类方法处理基础记忆
        await super().process_new_message(message, cancellation_token)
        
        if cancellation_token and cancellation_token._cancelled:
            return
        
        # 检查消息中是否包含座椅命令
        if isinstance(message.get('content'), str):
            content = message.get('content', '')
            
            # 检测座椅调整命令
            seat_command = self._extract_seat_command(content)
            if seat_command and self.seat_usage_history:
                # 将座椅命令添加到使用历史
                await self.add_to_seat_usage_history(
                    seat_position=seat_command.get('seat_position', ''),
                    command_type=seat_command.get('command_type', ''),
                    value=seat_command.get('value', ''),
                    user_id=message.get('metadata', {}).get('user_id', 'unknown')
                )
    
    def _extract_seat_command(self, text: str) -> Optional[Dict[str, Any]]:
        """
        从文本中提取座椅命令
        
        Args:
            text: 文本内容
            
        Returns:
            包含座椅命令信息的字典，如果没有找到则返回None
        """
        # 提取座椅位置
        seat_position = None
        for key, value in self.seat_positions.items():
            if key in text.lower() or value in text:
                seat_position = key
                break
        
        if not seat_position:
            return None
        
        # 提取命令类型
        command_type = None
        value = None
        
        # 检查按摩强度
        massage_intensity_match = re.search(r'按摩强度.*?(\d+)', text)
        if massage_intensity_match:
            command_type = "massage_intensity"
            value = int(massage_intensity_match.group(1))
        
        # 检查通风强度
        elif re.search(r'通风强度.*?(\d+)', text):
            command_type = "ventilation_intensity"
            value = int(re.search(r'通风强度.*?(\d+)', text).group(1))
        
        # 检查按摩模式
        elif '按摩模式' in text:
            command_type = "massage_mode"
            if '波浪' in text or 'wave' in text.lower():
                value = "wave"
            elif '脉冲' in text or 'pulse' in text.lower():
                value = "pulse"
        
        if command_type and value is not None:
            return {
                "seat_position": seat_position,
                "command_type": command_type,
                "value": value
            }
        
        return None
    
    async def extract_user_profile(self, message_history: List[Dict], cancellation_token: Optional[CancellationToken] = None) -> None:
        """
        从消息历史中提取用户座椅偏好
        
        分析用户的历史消息，识别用户对座椅的偏好并保存。
        
        Args:
            message_history: 历史消息列表
            cancellation_token: 可选的取消操作令牌
        """
        if cancellation_token and cancellation_token._cancelled:
            return
        
        if not self.user_profile or not self.llm_client:
            return
        
        # 收集用户的座椅指令
        seat_commands = []
        user_id = "unknown"
        
        for message in message_history:
            if message.get('role') == 'user':
                if message.get('metadata', {}).get('user_id'):
                    user_id = message.get('metadata', {}).get('user_id')
                
                command = self._extract_seat_command(message.get('content', ''))
                if command:
                    seat_commands.append(command)
        
        if not seat_commands:
            return
        
        # 使用LLM分析用户座椅偏好
        # 这里可以根据实际情况设计LLM的提示词
        prompt = f"""
        根据以下用户的座椅调整历史，提取用户对座椅的偏好：
        
        {seat_commands}
        
        请分析并总结用户对不同座位的偏好设置(按摩强度、通风强度、按摩模式)。
        回复格式为简洁的JSON格式。
        """
        
        try:
            result = await self.llm_client.generate_text_async(prompt=prompt)
            
            # 添加到用户画像
            profile_content = MemoryContent(
                content=result,
                mime_type=MemoryMimeType.TEXT,
                metadata={
                    "type": "seat_preference",
                    "user_id": user_id,
                    "timestamp": str(self._get_current_time())
                }
            )
            
            await self.add_to_user_profile(profile_content, cancellation_token)
        except Exception as e:
            print(f"提取用户座椅偏好时出错: {e}")
    
    async def add_to_seat_usage_history(self, seat_position: str, command_type: str, value: Any, user_id: str = "unknown", cancellation_token: Optional[CancellationToken] = None) -> None:
        """
        添加座椅使用记录到历史
        
        Args:
            seat_position: 座位位置
            command_type: 命令类型
            value: 设置值
            user_id: 用户ID
            cancellation_token: 可选的取消操作令牌
        """
        if not self.seat_usage_history or (cancellation_token and cancellation_token._cancelled):
            return
        
        # 构造使用记录内容
        seat_name = self.seat_positions.get(seat_position, seat_position)
        
        content_text = f"用户({user_id})将{seat_name}的{command_type}设置为{value}"
        
        usage_content = MemoryContent(
            content=content_text,
            mime_type=MemoryMimeType.TEXT,
            metadata={
                "seat_position": seat_position,
                "command_type": command_type,
                "value": value,
                "user_id": user_id,
                "timestamp": str(self._get_current_time())
            }
        )
        
        await self.seat_usage_history.add(usage_content, cancellation_token)
        
        # 更新用户座椅偏好
        if self.seat_preference:
            # 创建或更新用户对该座椅的偏好记录
            preference_key = f"{user_id}_{seat_position}"
            
            # 查询现有偏好
            existing_preference = await self.seat_preference.query(preference_key)
            
            preference_data = {}
            if existing_preference.results:
                # 更新现有记录
                try:
                    preference_data = eval(existing_preference.results[0].content)
                except:
                    preference_data = {}
            
            # 更新对应的设置值
            preference_data[command_type] = value
            preference_data['user_id'] = user_id
            preference_data['seat_position'] = seat_position
            preference_data['last_updated'] = str(self._get_current_time())
            
            # 保存更新后的偏好
            preference_content = MemoryContent(
                content=str(preference_data),
                mime_type=MemoryMimeType.TEXT,
                metadata={
                    "key": preference_key,
                    "type": "seat_preference"
                }
            )
            
            await self.seat_preference.add(preference_content, cancellation_token)
    
    async def save_scene_mode(self, scene_name: str, seat_settings: Dict[str, Dict[str, Any]], description: str = "", cancellation_token: Optional[CancellationToken] = None) -> None:
        """
        保存场景模式
        
        保存特定场景下的多个座椅设置组合
        
        Args:
            scene_name: 场景名称
            seat_settings: 座椅设置，格式为 {seat_position: {setting_type: value}}
            description: 场景描述
            cancellation_token: 可选的取消操作令牌
        """
        if not self.scene_mode_memory or (cancellation_token and cancellation_token._cancelled):
            return
        
        # 构造场景内容
        scene_text = f"场景模式: {scene_name}\n描述: {description}\n设置:\n"
        
        for seat_position, settings in seat_settings.items():
            seat_name = self.seat_positions.get(seat_position, seat_position)
            scene_text += f"- {seat_name}: "
            
            setting_texts = []
            for setting_type, value in settings.items():
                setting_texts.append(f"{setting_type}={value}")
            
            scene_text += ", ".join(setting_texts) + "\n"
        
        scene_content = MemoryContent(
            content=scene_text,
            mime_type=MemoryMimeType.TEXT,
            metadata={
                "scene_name": scene_name,
                "settings": seat_settings,
                "description": description,
                "timestamp": str(self._get_current_time())
            }
        )
        
        await self.scene_mode_memory.add(scene_content, cancellation_token)
    
    async def get_user_seat_preference(self, user_id: str, seat_position: str, cancellation_token: Optional[CancellationToken] = None) -> Optional[Dict[str, Any]]:
        """
        获取用户对特定座椅的偏好设置
        
        Args:
            user_id: 用户ID
            seat_position: 座位位置
            cancellation_token: 可选的取消操作令牌
            
        Returns:
            用户座椅偏好设置字典，如果未找到则返回None
        """
        if not self.seat_preference or (cancellation_token and cancellation_token._cancelled):
            return None
        
        preference_key = f"{user_id}_{seat_position}"
        result = await self.seat_preference.query(preference_key, cancellation_token)
        
        if result.results:
            try:
                return eval(result.results[0].content)
            except:
                return None
        
        return None
    
    def _get_current_time(self) -> float:
        """获取当前时间戳"""
        import asyncio
        return asyncio.get_event_loop().time()
