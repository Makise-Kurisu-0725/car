import os
import json
import dotenv
from agents.agents import *
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

dotenv.load_dotenv()

import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from fastapi import APIRouter

# 初始化设备管理器
manager = DeviceManager()

router = APIRouter()

@router.get("/environment-info")
async def get_environment_info():
    return manager.collect_environment_info()


async def main():
    # 创建座椅Agent
    seat_agent = SeatAgent(
        manager, 
        name="seat_agent", 
        description="座椅智能体。可以获取与设置特定位置座椅的信息。包括座椅按摩模式，按摩档位、座椅加热、座椅通风模式、座椅占位信息。"
    )
    
    # 创建美团Agent
    meituan_agent = MeituanAgent(
        manager, 
        name="meituan_agent", 
        description="美团外卖订餐智能体。可以搜索不同类型的商家，查询菜单，点餐(将菜品加入购物车)，设置配送地点，以及完成支付。"
    )

    # 创建音视频Agent
    av_agent = AVAgent(
        manager, 
        name="audio_agent", 
        description="音视频智能体。可以控制车内音乐播放器和视频播放器。"
    )

    # 创建出行Agent
    navigation_agent = NavigationAgent(
        manager, 
        name="navigation_agent", 
        description="出行智能体。可以搜索导航路线。"
    )

    # 创建用车Agent
    use_agent = UseAgent(
        manager, 
        name="use_agent", 
        description="用车智能体。可以监测车内温度、GPS位置、摄像头识别乘客、车内声音，以及提供轮胎、电池等硬件状况查询功能。"
    )

    # 创建理想同学
    Li = LiAgent(
        name="Li",
        agent_map={
            "seat_agent": seat_agent,
            "meituan_agent": meituan_agent,
            "av_agent":av_agent,
            "navigation_agent":navigation_agent,
            "use_agent":use_agent
        },
        description="多智能体调度中心，任务的分解者",
        device_manager = manager
    )

    data = manager.collect_environment_info()

    # with open("device_state.json", 'w', encoding='utf-8') as f:
    #     data = json.loads(data)
    #     json.dump(data, f, 
    #                 ensure_ascii=False,  # 保留中文原样
    #                 indent=2)      
    # query_list = [
    #     #负例： {"content":"请将左后门的乘客座椅的运动模式打开","agent":"座椅agent"},  
    #     {"content":"请将左后门的乘客按摩强度调到最高，并将主驾的座椅通风打开","agent":"座椅agent"},
    #     {"content":"帮我找一家烧烤店，查看菜单后点一个烤羊肉，然后送到海淀区中关村大街1号，直接付款","agent":"美团agent"},
    #     {"content":"请播一首陈奕迅的十面埋伏","agent":"音视频agent"},
    #     {"content":"从公司导航到家","agent":"出行agent"},
    #     {"content":"现在车里多少度啊","agent":"用车agent"}
    # ]
    # for i  in query_list:
    #     #执行座椅测试任务
    #     print(f"=== {i['agent']}功能测试 ===")
        
    #     plan = await Li.on_messages([TextMessage(content=i["content"], source='user')], cancellation_token=CancellationToken())
        
    #     if plan.chat_message.content[0] == '错':
    #         print(plan.chat_message.content)
    #     else:
    #         print("理想同学的规划是：", json.loads(plan.chat_message.content))
    #         try:
    #             await Li.execute_plan(plan.chat_message.content, CancellationToken())
    #             print("执行完毕")
    #         except asyncio.TimeoutError:
    #             print("执行超时已取消")
    #     # await av_agent.on_messages([TextMessage(content="请播一首陈奕迅的十面埋伏",source="Li")],CancellationToken())
                
if __name__ == "__main__":
    asyncio.run(main())
