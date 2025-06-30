import asyncio
import json
from typing import Sequence, Dict

from autogen_core import CancellationToken
from autogen_agentchat.base import Response
from autogen_agentchat.agents import BaseChatAgent
from autogen_core.tools import FunctionTool
from autogen_agentchat.messages import TextMessage, ChatMessage

from baseagent import BaseAgent
from utils import *
from device import *
from tools import *
class SeatAgent(BaseAgent):
    """智能座椅控制代理"""

    def __init__(self, device_manager: DeviceManager, name:str,description:str, **kwargs):
        device_type = [Seat]
        self.device_manager = device_manager
        super().__init__(name=name,description=description,device_type=device_type,**kwargs)
        self._register_device()
        self._generate_func_description()


    def _register_device(self):
        self.device_manager.register_device(Seat(Seat.Position.DRIVER))
        self.device_manager.register_device(Seat(Seat.Position.PASSENGER))
        self.device_manager.register_device(Seat(Seat.Position.REAR_LEFT))
        self.device_manager.register_device(Seat(Seat.Position.REAR_RIGHT))
    
    def _generate_func_description(self):

        seats_id = self.device_manager.get_type_devices_ids(Seat)
        self._tools=[
        FunctionTool(adjust_massage_intensity_wrapper(self.device_manager), description=f"调整座椅的按摩强度，需要提供座椅的id和按摩强度，座椅id可以从{seats_id}中选择"),
        FunctionTool(adjust_ventilation_intensity_wrapper(self.device_manager), description=f"调整座椅的通风强度，需要提供座椅的id和通风强度，座椅id可以从{seats_id}中选择"),
        FunctionTool(adjust_massage_mode_wrapper(self.device_manager), description=f"调整座椅的按摩模式，需要提供座椅的id和按摩模式，座椅id可以从{seats_id}中选择。按摩的模式有两个可用值，其中`wave`代表波浪按摩，`pulse`代表脉冲按摩")
    ]


class AVAgent(BaseAgent):
    """音视频智能体"""

    def __init__(self, device_manager: DeviceManager, name:str,description:str, **kwargs):
        device_type = [MusicPlayer,VideoPlayer]
        self.device_manager = device_manager
        super().__init__(name=name,description=description,device_type=device_type,**kwargs)
        self._register_device()
        self._generate_func_description()


    def _register_device(self):
        self.device_manager.register_device(MusicPlayer())
        self.device_manager.register_device(VideoPlayer())
    
    
    def _generate_func_description(self):
        music_player_id = self.device_manager.get_type_devices_ids(MusicPlayer)[0]
        video_player_id = self.device_manager.get_type_devices_ids(VideoPlayer)[0]
        self._tools=[
                FunctionTool(play_music_wrapper(self.device_manager), description=f"播放音乐，其中音乐播放器的设备ID为{music_player_id}"),
                FunctionTool(play_video_wrapper(self.device_manager), description=f"播放视频，其中视频播放器的设备ID为{video_player_id}"),
        ]

class UseAgent(BaseAgent):
    """用车智能体"""

    def __init__(self, device_manager: DeviceManager, name:str,description:str, **kwargs):
        device_type = [Temperature_device,Tire,Timeclock,Battery,GPSDevice,Camera,Voice]
        self.device_manager = device_manager
        super().__init__(name=name,description=description,device_type=device_type,**kwargs)
        self._register_device()
        self._generate_func_description()


    def _register_device(self):
        self.device_manager.register_device(Temperature_device(Temperature_device.Position.INSIDE))
        self.device_manager.register_device(Tire(Tire.Position.LEFT_FRONT))
        self.device_manager.register_device(Timeclock())
        self.device_manager.register_device(Battery())
        self.device_manager.register_device(GPSDevice())
        self.device_manager.register_device(Camera(Camera.Position.DRIVER))
        self.device_manager.register_device(Voice())
    
    
    def _generate_func_description(self):
        Temperature_device_ID = self.device_manager.get_type_devices_ids(Temperature_device)
        Tire_ID = self.device_manager.get_type_devices_ids(Tire)
        Timeclock_ID = self.device_manager.get_type_devices_ids(Timeclock)
        Battery_ID = self.device_manager.get_type_devices_ids(Battery)
        GPSDevice_ID = self.device_manager.get_type_devices_ids(GPSDevice)
        Camera_ID = self.device_manager.get_type_devices_ids(Camera)
        Voice_ID = self.device_manager.get_type_devices_ids(Voice)

        self._tools=[
            FunctionTool(get_Cabin_Temperature_wrapper(self.device_manager), description=f"查询当前车内的温度，其中温度设备ID为{Temperature_device_ID}"),
            FunctionTool(get_Tire_Status_wrapper(self.device_manager), description=f"查询轮胎状态，需要提供轮胎的id和查询种类参数,查询种类参数从以下两种进行选择:[胎压，胎温]，轮胎的id可以从{Tire_ID}中选择"),
            FunctionTool(get_CurrentTime_wrapper(self.device_manager), description=f"获取当前时间，其中时钟设备ID为{Timeclock_ID}"),
            FunctionTool(get_SOC_wrapper(self.device_manager), description=f"查询车辆剩余电量，其中电池设备ID为{Battery_ID}"),
            FunctionTool(get_GPS_Location_wrapper(self.device_manager), description=f"查询车辆GPS坐标，其中GPS定位设备ID为{GPSDevice_ID}"),
            FunctionTool(get_Passager_wrapper(self.device_manager), description=f"获取当前摄像头下的用户面容ID，需要提供摄像头的设备id，摄像头的id可以从{Camera_ID}中选择"),
            FunctionTool(get_Voice_wrapper(self.device_manager), description=f"获取当前说话人的声音ID，其中声音设备ID为{Voice_ID}")
        ]

 
class MeituanAgent(BaseAgent):
    """美团智能体"""

    def __init__(self, device_manager: DeviceManager, name:str,description:str, **kwargs):
        device_type = [MeituanApp]
        self.device_manager = device_manager
        super().__init__(name=name,description=description,device_type=device_type,**kwargs)
        self._register_device()
        self._generate_func_description()


    def _register_device(self):
        self.device_manager.register_device(MeituanApp())
    
    def _generate_func_description(self):
        self._tools=[
        FunctionTool(search_shops_by_type_wrapper(self.device_manager), description="按商家类型搜索，可选类型包括：'烧烤'、'快餐'、'面食'、'川菜'、'粤菜'"),
        FunctionTool(get_shop_menu_wrapper(self.device_manager), description="获取指定商家ID的菜单，例如'S001'"),
        FunctionTool(add_to_cart_wrapper(self.device_manager), description="将用户点的菜品加入购物车"),
        FunctionTool(add_delivery_location_wrapper(self.device_manager), description="为订单添加配送地址"),
        FunctionTool(process_payment_wrapper(self.device_manager), description="处理订单支付，可指定支付方式")
        ]

class NavigationAgent(BaseAgent):
    """出行智能体"""

    def __init__(self, device_manager: DeviceManager, name:str,description:str, **kwargs):
        device_type = [Navigator]
        self.device_manager = device_manager
        super().__init__(name=name,description=description,device_type=device_type,**kwargs)
        self._register_device()
        self._generate_func_description()


    def _register_device(self):
        self.device_manager.register_device(Navigator())
    
    def _generate_func_description(self):
        self._tools=[
        FunctionTool(get_path_wrapper(self.device_manager), description="可搜索从起点到终点的路径,需要提供起点和终点的位置")
        ]


from pydantic import BaseModel
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.base import Response
from autogen_core import CancellationToken

class LiAgent(BaseAgent):
    def __init__(
        self,
        name:str,
        description:str,
        agent_map: Dict[str, BaseChatAgent],
        device_manager: DeviceManager,
        **kwargs
    ):
        device_type = None
        self.device_manager = device_manager
        super().__init__( 
            system_message=self._create_system_prompt(agent_map),
            name=name,
            description=description,
            device_type = device_type,
            **kwargs
        )
        self.agent_map = agent_map
        self.active_tasks = set()
        self.semaphore = asyncio.Semaphore(5)  # 最大并行任务数

    def _create_system_prompt(self, agents: Dict) -> str:
        agents_desc = "\n".join([f"- {name}: {agent.description}" 
                               for name, agent in agents.items()])
        return f"""## 你是理想汽车智能驾舱的多智能体任务分解专家，擅长扩充用户模糊的需求，并分解成多个简短的、包含单个指令的任务。
你需要依据车内智能体的功能信息，将用户的需求分解为多个由边缘智能体执行的任务，并输出指定格式的json。
你分解出的任务必须明确具体（如具体是哪个座位，修改为什么档位），可能需要对用户的需求进行一些扩充。

边缘智能体：
{agents_desc}

输出要求：
1. 只能返回格式正确的JSON，我需要用json.loads直接读取你返回的信息。每个任务包含agent、content、parallel、sequence四个字段。
2. agent必须属于[边缘智能体]。
3. content为其他智能体可以执行的任务。尽可能简洁，同时遵循[可用智能体]中的功能描述。如果用户的需求含糊不清或不具体，例如未指定具体的参数，请自行补充。
4. parallel字段为true表示任务可与下一个任务并行。请仔细思考用户的需求中是否存在先后关系，对于存在先后关系的任务则parallel字段为false。
5. sequence表示任务执行顺序。
6. 边缘智能体能够一次性执行多个任务。你需要尽可能将属于一个智能体的任务划分到一次AgentCall中。

输出示例：
[
    {{
        "agent": "AVagent",
        "content": "播放周杰伦的歌，声音设置为50%，并把这首歌加入收藏",
        "parallel": true,
        "sequence": 1
    }},
    {{
        "agent": "AirconditionAgent",
        "content": "打开主驾空调",
        "parallel": true,
        "sequence": 2
    }}
]

用户需求:
"""
    async def on_messages(
        self,
        messages: Sequence[ChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        """处理输入消息并生成任务计划"""
        self.cancellation_token = cancellation_token
        try:
            plan = await self._generate_task_plan(messages,cancellation_token)
            return Response(
                chat_message=self._create_task_message(plan)
            )
        except Exception as e:
            return Response(
                chat_message=TextMessage(content=f"错误: {str(e)}", source=self.name)
            )

    async def _generate_task_plan(self, messages: Sequence[ChatMessage],cancellation_token:CancellationToken) -> list:
        """生成带并行标记的任务计划"""
        #messages = self._add_environment_info(messages)
        response = await super().on_messages(messages, cancellation_token)
        self._clean_environment_messages()
        response = self._parse_plan(response)
        return response
    
    def _parse_plan(self,response:Response)->list:
        response = response.chat_message.content
        begin_pos = response.find('[')
        end_pos = response.find(']')
        plan_str = response[begin_pos:end_pos+1]
        if begin_pos!=-1:
            plan = json.loads(plan_str)
            return plan
        else:
            raise ValueError("未产生json plan")

    def _create_task_message(self, plan: list) -> ChatMessage:
        """创建任务执行消息"""
        return TextMessage(
            content=json.dumps(plan),
            metadata={"type": "task_plan"},
            source=self.name
        )

    async def execute_plan(
        self,
        plan: str,
        cancellation_token: CancellationToken  # 添加参数
    ) -> bool:
        """执行任务计划"""
        plan = json.loads(plan)
        sorted_tasks = sorted(plan, key=lambda x: x["sequence"])
        signal = True

        for task in sorted_tasks:
            if task["parallel"]:
                asyncio.create_task(
                    self._execute_parallel_task(task, cancellation_token)  # 传递令牌
                )
            else:
                result = await self._execute_sequential_task(task, cancellation_token)  # 传递令牌
                # if not (result is True and signal is True):
                #     signal = False
        return signal
    
    async def _execute_parallel_task(
        self, 
        task: Dict,
        cancellation_token: CancellationToken  # 添加参数
    ):
        """执行并行任务"""
        async with self.semaphore:
            await self._execute_task(task, cancellation_token)

    async def _execute_sequential_task(
        self, 
        task: Dict,
        cancellation_token: CancellationToken  # 添加参数
    ):
        """执行串行任务"""
        if not await self._check_prerequisites(task):
            print(f"前置任务未完成，已跳过: {task['agent']}")
            return False
            
        result = await self._execute_task(task, cancellation_token)  # 传递令牌
        if not result:
            print(f"任务失败已跳过: {task['agent']}")
            return False

    async def _execute_task(
        self, 
        task: Dict,
        cancellation_token: CancellationToken  # 添加参数
    ) -> bool:
        """实际任务执行逻辑"""
        try:
            agent = self.agent_map[task["agent"]]
            response = await agent.on_messages(
                [TextMessage(content=task["content"], source=self.name)],
                cancellation_token  # 传递到子智能体
            )
            
            success = "success" in response.chat_message.metadata["label"]
            return success
        # except (asyncio.CancelledError, CancellationToken.is_cancelled()):
        #     print(f"任务已取消: {task['agent']}")
        #     return False
        except KeyError:
            print(f"智能体未注册: {task['agent']}")
            return False

    async def _check_prerequisites(self, task: Dict) -> bool:
        """检查前置任务状态"""
        # 实现前置依赖检查逻辑
        return True
    

