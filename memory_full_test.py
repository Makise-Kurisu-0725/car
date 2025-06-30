import os
os.environ["API_KEY"] = "sk-xxxxxxxxxx"
import asyncio
import json
from typing import List, Dict, Any, Optional
import argparse

from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core._cancellation_token import CancellationToken
from autogen_agentchat.messages import TextMessage

from agents.basememoryagent import BaseMemoryAgent
from memory_module.base_agent_memory import BaseAgentMemoryConfig

from device import DeviceManager
manager = DeviceManager()
from fastapi import APIRouter
router = APIRouter()
@router.get("/environment-info")
async def get_environment_info():
    return manager.collect_environment_info()
# 测试数据目录
TEST_DATA_DIR = "./data/memory_test"
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# 默认记忆路径
DEFAULT_MEMORY_PATH = os.path.join(TEST_DATA_DIR, "default_memory")


# 模拟虚拟对话数据
MOCK_CONVERSATIONS = [
    {
        "role": "user",
        "content": "你好，我叫张三，我想了解一下自动驾驶汽车的工作原理"
    },
    {
        "role": "assistant",
        "content": "你好张三！自动驾驶汽车主要依靠传感器、计算机视觉和AI算法来感知环境并做出决策。它们使用激光雷达、摄像头和雷达等传感器来检测周围的物体，然后通过AI算法分析这些数据来控制车辆的方向、速度和制动。"
    },
    {
        "role": "user",
        "content": "自动驾驶汽车的安全性如何？"
    },
    {
        "role": "assistant",
        "content": "自动驾驶汽车的安全性是个复杂话题。一方面，它们不会疲劳、分心或情绪化驾驶，这减少了人为错误；另一方面，当前技术在复杂或罕见场景下可能面临挑战。大多数研究表明，成熟的自动驾驶技术有潜力大幅减少交通事故，但完全安全仍需要技术进一步发展和完善的法规框架。"
    },
    {
        "role": "user",
        "content": "我对电动汽车也很感兴趣，特别是理想的L9"
    },
    {
        "role": "assistant",
        "content": "理想L9是一款高端电动SUV，采用增程式电动系统，结合了电动车的环保性和传统燃油车的续航能力。它提供约1100公里的综合续航里程，搭载先进的智能驾驶辅助系统。L9空间宽敞、内饰豪华，配备了大尺寸中控屏和智能座舱系统，性价比较高。作为理想旗下的旗舰车型，L9在市场上受到不少关注。"
    }
]

# 更多虚拟对话，用于测试记忆检索
ADDITIONAL_CONVERSATIONS = [
    {
        "role": "user",
        "content": "我想购买一辆SUV类型的电动汽车，有什么推荐吗？"
    },
    {
        "role": "assistant", 
        "content": "考虑到你对电动汽车的兴趣，我可以推荐几款电动SUV：理想L9是一个不错的选择，它提供良好的性能和空间表现；比亚迪唐EV也是国产电动SUV中的佼佼者；如果预算充足，奥迪e-tron和奔驰EQC这样的豪华电动SUV也值得考虑。这些车型都提供了不错的空间、续航里程和智能驾驶辅助功能。"
    },
    {
        "role": "user",
        "content": "我记得你说过理想有自动驾驶功能，L9的自动驾驶功能如何？"
    },
    {
        "role": "assistant",
        "content": "是的，张三，理想L9确实配备了先进的智能驾驶辅助系统，称为NOA（导航辅助驾驶）。这个系统包括自适应巡航控制、车道保持、自动变道、自动超车等功能。L9搭载了6颗8百万像素摄像头、5个毫米波雷达和其他传感器，能够在高速公路和城市道路上提供良好的辅助驾驶体验。不过需要注意的是，即使启用了这些功能，驾驶员仍需保持注意力并随时准备接管车辆控制。"
    }
]

# 模拟用户画像数据
MOCK_USER_PROFILE = [
    {
        "name": "张三",
        "interests": ["自动驾驶", "电动汽车", "理想"],
        "preferences": {
            "car_types": ["电动车", "SUV"],
            "brands": ["理想", "比亚迪"]
        }
    }
]

# 模拟任务反思数据
MOCK_REFLECTIONS = [
    {
        "task": "解释自动驾驶汽车的工作原理",
        "reflection": "用户张三询问了自动驾驶汽车的工作原理，我提供了关于传感器、计算机视觉和AI算法如何协同工作的简明解释。用户似乎对自动驾驶和电动车技术都很感兴趣，特别是对理想品牌有偏好。在未来交流中，可以更多地关注自动驾驶的技术细节和电动车品牌对比，尤其是理想的产品线。"
    },
    {
        "task": "讨论电动SUV选择",
        "reflection": "用户张三询问了电动SUV的推荐，我根据他之前表现出的对理想的兴趣，优先推荐了L9，同时也提供了其他几款不同价位的选择。用户还特别关注自动驾驶功能，表明这可能是他购车决策的重要因素。在未来互动中，可以更详细地介绍各款车型的自动驾驶能力对比。"
    }
]


def memory_exists(path: str) -> bool:
    """检查指定路径是否存在记忆数据"""
    # 检查关键记忆文件或目录是否存在
    vector_memory_path = os.path.join(path, "reflection_memory")
    dict_memory_path = os.path.join(path, "short_term_memory.json")
    user_profile_path = os.path.join(path, "user_profile.json")
    
    # 只要有一个关键记忆组件存在，就认为记忆存在
    return (os.path.exists(vector_memory_path) or 
            os.path.exists(dict_memory_path) or 
            os.path.exists(user_profile_path))


async def convert_to_chat_messages(conversations: List[Dict[str, str]]) -> List[TextMessage]:
    """将对话字典列表转换为TextMessage列表"""
    chat_messages = []
    for message in conversations:
        chat_message = TextMessage(
            content=message["content"],
            source=message["role"]
        )
        chat_messages.append(chat_message)
    return chat_messages


async def test_memory_functionality():
    """测试记忆功能的基本操作"""
    print("\n========== 测试记忆功能的基本操作 ==========")
    
    # 创建一个测试用的记忆配置
    memory_config = BaseAgentMemoryConfig(
        name="test_memory",
        base_path=os.path.join(TEST_DATA_DIR, "test_memory_1")
    )
    
    # 创建一个测试用的智能体
    agent = BaseMemoryAgent(
        name="test_agent",
        description="用于测试记忆功能的智能体",
        device_type="virtual",
        memory_config=memory_config,
        memory_base_path=TEST_DATA_DIR
    )
    
    # 创建取消令牌
    cancellation_token = CancellationToken()
    
    # 1. 测试添加记忆
    print("\n--- 测试添加对话记忆 ---")
    for message in MOCK_CONVERSATIONS:
        await agent.agent_memory.process_new_message(message, cancellation_token)
    print("✓ 成功添加模拟对话数据")
    
    # 2. 测试添加用户画像
    print("\n--- 测试添加用户画像 ---")
    for profile in MOCK_USER_PROFILE:
        content = MemoryContent(
            content=json.dumps(profile, ensure_ascii=False),
            mime_type=MemoryMimeType.JSON,
            metadata={"type": "user_profile"}
        )
        await agent.agent_memory.add_to_user_profile(content, cancellation_token)
    print("✓ 成功添加用户画像数据")
    
    # 3. 测试自动生成任务反思（而不是使用模拟数据）
    print("\n--- 测试自动生成任务反思 ---")
    # 为第一个对话生成反思
    first_interaction = [
        MOCK_CONVERSATIONS[0],  # 用户询问自动驾驶
        MOCK_CONVERSATIONS[1]   # 助手回答
    ]
    reflection = await agent.agent_memory.generate_reflection(
        task_description="解释自动驾驶汽车的工作原理",
        message_history=first_interaction,
        cancellation_token=cancellation_token
    )
    print(f"✓ 成功生成反思: {reflection[:100]}..." if len(reflection) > 100 else f"✓ 成功生成反思: {reflection}")
    
    # 4. 测试记忆检索 - 对话记忆
    print("\n--- 测试检索对话记忆 ---")
    conv_results = await agent.agent_memory.query_conversation_memory("自动驾驶安全性", cancellation_token)
    print(f"找到 {len(conv_results.results)} 条相关对话记忆")
    for i, memory in enumerate(conv_results.results, 1):
        print(f"记忆 {i}: {memory.content[:100]}..." if len(memory.content) > 100 else f"记忆 {i}: {memory.content}")
    
    # 5. 测试记忆检索 - 用户画像
    print("\n--- 测试检索用户画像 ---")
    profile_results = await agent.agent_memory.query_user_profile("用户兴趣", cancellation_token)
    print(f"找到 {len(profile_results.results)} 条用户画像信息")
    for i, memory in enumerate(profile_results.results, 1):
        print(f"画像 {i}: {memory.content}")
    
    # 6. 测试记忆检索 - 任务反思
    print("\n--- 测试检索任务反思 ---")
    reflection_results = await agent.agent_memory.query_reflection_memory("电动汽车", cancellation_token)
    print(f"找到 {len(reflection_results.results)} 条相关反思")
    for i, memory in enumerate(reflection_results.results, 1):
        print(f"反思 {i}: {memory.content[:100]}..." if len(memory.content) > 100 else f"反思 {i}: {memory.content}")
    
    # 关闭记忆系统
    await agent.agent_memory.close()
    
    print("\n✓ 基本记忆功能测试完成")


async def test_memory_persistence():
    """测试记忆的持久化和加载"""
    print("\n========== 测试记忆的持久化和加载 ==========")
    
    memory_path = os.path.join(TEST_DATA_DIR, "persistent_memory")
    
    # 1. 创建并保存记忆
    print("\n--- 创建并保存记忆 ---")
    
    # 创建记忆配置
    memory_config = BaseAgentMemoryConfig(
        name="persistent_memory",
        base_path=memory_path
    )
    
    # 创建智能体
    agent1 = BaseMemoryAgent(
        name="agent_with_memory",
        description="用于测试记忆持久化的智能体",
        device_type="virtual",
        memory_config=memory_config
    )
    
    # 取消令牌
    cancellation_token = CancellationToken()
    
    # 添加一些记忆数据
    for message in MOCK_CONVERSATIONS:
        await agent1.agent_memory.process_new_message(message, cancellation_token)
    
    # 添加用户画像
    for profile in MOCK_USER_PROFILE:
        content = MemoryContent(
            content=json.dumps(profile, ensure_ascii=False),
            mime_type=MemoryMimeType.JSON,
            metadata={"type": "user_profile"}
        )
        await agent1.agent_memory.add_to_user_profile(content, cancellation_token)
    
    # 使用记忆模块的反思生成功能生成反思
    print("\n--- 生成任务反思 ---")
    first_task = "解释自动驾驶汽车的工作原理"
    first_history = [MOCK_CONVERSATIONS[0], MOCK_CONVERSATIONS[1]]
    
    reflection1 = await agent1.agent_memory.generate_reflection(
        task_description=first_task,
        message_history=first_history,
        cancellation_token=cancellation_token
    )
    
    print(f"生成的反思: {reflection1[:100]}..." if reflection1 and len(reflection1) > 100 else f"生成的反思: {reflection1}")
    
    # 关闭记忆系统，触发持久化
    await agent1.agent_memory.close()
    
    print(f"✓ 记忆已保存到: {memory_path}")
    
    # 2. 创建新智能体，加载已保存的记忆
    print("\n--- 加载已保存的记忆 ---")
    
    # 创建同样配置的记忆系统
    memory_config2 = BaseAgentMemoryConfig(
        name="persistent_memory",
        base_path=memory_path
    )
    
    # 创建新智能体
    agent2 = BaseMemoryAgent(
        name="agent_loading_memory",
        description="用于测试记忆加载的智能体",
        device_type="virtual",
        memory_config=memory_config2
    )
    
    # 测试检索以验证记忆加载成功
    user_results = await agent2.agent_memory.query_user_profile("张三", cancellation_token)
    conv_results = await agent2.agent_memory.query_conversation_memory("理想", cancellation_token)
    reflection_results = await agent2.agent_memory.query_reflection_memory(first_task, cancellation_token)
    
    print(f"✓ 从保存的记忆中检索到 {len(user_results.results)} 条用户画像信息")
    print(f"✓ 从保存的记忆中检索到 {len(conv_results.results)} 条关于理想汽车的对话记忆")
    print(f"✓ 从保存的记忆中检索到 {len(reflection_results.results)} 条相关反思")
    
    # 关闭记忆系统
    await agent2.agent_memory.close()
    
    print("\n✓ 记忆持久化和加载测试完成")


async def test_memory_query_comparison():
    """测试有无记忆的回答区别"""
    print("\n========== 测试有无记忆的回答区别 ==========")
    
    # 准备测试查询
    test_queries = [
        "我想买一辆电动汽车，有什么推荐？",
        "我之前对什么车型表示过兴趣？",
        "我是谁？我对什么感兴趣？"
    ]
    
    # 1. 没有记忆的智能体
    print("\n--- 没有记忆的智能体 ---")
    
    no_memory_config = BaseAgentMemoryConfig(
        name="no_memory",
        base_path=os.path.join(TEST_DATA_DIR, "no_memory"),
        use_memory=False
    )
    
    no_memory_agent = BaseMemoryAgent(
        name="agent_without_memory",
        device_manager=manager,
        description="没有记忆的测试智能体",
        device_type="virtual",
        memory_config=no_memory_config,
    )
    
    cancellation_token = CancellationToken()
    
    print("查询及真实回答 (无记忆):")
    for query in test_queries:
        print(f"\n查询: {query}")
        
        # 创建查询消息
        query_message = TextMessage(content=query, source="user")
        
        # 实际调用智能体生成回答
        response = await no_memory_agent.on_messages([query_message], cancellation_token)
        
        # 打印回答
        print(f"智能体回答: {response.chat_message.content}")
    
    await no_memory_agent.agent_memory.close()
    
    # 2. 有记忆的智能体
    print("\n--- 有记忆的智能体 ---")
    
    memory_path = os.path.join(TEST_DATA_DIR, "with_memory")
    
    with_memory_config = BaseAgentMemoryConfig(
        name="with_memory",
        base_path=memory_path
    )
    
    with_memory_agent = BaseMemoryAgent(
        name="agent_with_memory",
        device_manager=manager,
        description="有记忆的测试智能体",
        device_type="virtual",
        memory_config=with_memory_config
    )
    
    # 添加记忆数据
    for message in MOCK_CONVERSATIONS + ADDITIONAL_CONVERSATIONS:
        await with_memory_agent.agent_memory.process_new_message(message, cancellation_token)
    
    # 添加用户画像
    for profile in MOCK_USER_PROFILE:
        content = MemoryContent(
            content=json.dumps(profile, ensure_ascii=False),
            mime_type=MemoryMimeType.JSON,
            metadata={"type": "user_profile"}
        )
        await with_memory_agent.agent_memory.add_to_user_profile(content, cancellation_token)
    
    # 使用实际的记忆模块生成反思
    for i in range(0, len(MOCK_CONVERSATIONS), 2):
        if i+1 < len(MOCK_CONVERSATIONS):
            conversation_pair = [MOCK_CONVERSATIONS[i], MOCK_CONVERSATIONS[i+1]]
            task = conversation_pair[0]["content"]
            await with_memory_agent.agent_memory.generate_reflection(
                task_description=task,
                message_history=conversation_pair,
                cancellation_token=cancellation_token
            )
    
    print("查询及真实回答 (有记忆):")
    for query in test_queries:
        print(f"\n查询: {query}")
        
        # 创建查询消息
        query_message = TextMessage(content=query, source="user")
        
        # 实际调用智能体生成回答
        response = await with_memory_agent.on_messages([query_message], cancellation_token)
        
        # 打印回答
        print(f"智能体回答: {response.chat_message.content}")
    
    await with_memory_agent.agent_memory.close()
    
    print("\n✓ 有无记忆的回答对比测试完成")


async def create_default_memory():
    """创建并保存默认初始记忆"""
    print("\n========== 创建并保存默认初始记忆 ==========")
    
    # 如果默认记忆已存在，询问是否覆盖
    if memory_exists(DEFAULT_MEMORY_PATH):
        print(f"默认记忆已存在于: {DEFAULT_MEMORY_PATH}")
        if not args.force:
            print("跳过创建默认记忆，使用--force选项可以强制覆盖")
            return
        else:
            print("准备覆盖已有默认记忆")
    
    # 创建记忆配置
    memory_config = BaseAgentMemoryConfig(
        name="default_memory",
        base_path=DEFAULT_MEMORY_PATH
    )
    
    # 创建智能体
    default_agent = BaseMemoryAgent(
        name="default_agent",
        description="默认初始记忆智能体",
        device_type="virtual",
        memory_config=memory_config
    )
    
    cancellation_token = CancellationToken()
    
    # 创建一组默认记忆
    
    # 1. 添加一些常见问题的反思
    default_reflections = [
        {
            "task": "解释自动驾驶等级",
            "history": [
                {"role": "user", "content": "自动驾驶汽车有哪些等级？"},
                {"role": "assistant", "content": "自动驾驶分为L0到L5六个等级，从无自动化到完全自动化。"}
            ]
        },
        {
            "task": "比较不同电动汽车品牌",
            "history": [
                {"role": "user", "content": "市面上主要的电动汽车品牌有哪些？各有什么特点？"},
                {"role": "assistant", "content": "主要电动汽车品牌包括特斯拉、比亚迪、蔚来、小鹏、理想等，各有不同特点和优势。"}
            ]
        },
        {
            "task": "讨论电动汽车充电基础设施",
            "history": [
                {"role": "user", "content": "中国电动汽车的充电基础设施现状如何？"},
                {"role": "assistant", "content": "中国拥有全球最大的充电网络，但分布不均衡，一线城市密集而偏远地区稀少。"}
            ]
        }
    ]
    
    print("生成反思记忆...")
    for item in default_reflections:
        reflection = await default_agent.agent_memory.generate_reflection(
            task_description=item["task"],
            message_history=item["history"],
            cancellation_token=cancellation_token
        )
        print(f"✓ 为任务 '{item['task']}' 生成反思: {reflection[:50]}..." if reflection else "× 反思生成失败")
    
    # 2. 添加一些常见知识到短期记忆
    common_knowledge = [
        {
            "topic": "电动汽车电池类型",
            "content": "电动汽车常用电池类型包括锂离子电池、磷酸铁锂电池、固态电池等。锂离子电池能量密度高但安全性较低；磷酸铁锂电池安全性高但能量密度较低；固态电池是未来发展方向，具有高能量密度和高安全性。"
        },
        {
            "topic": "自动驾驶传感器",
            "content": "自动驾驶汽车主要使用的传感器包括激光雷达、毫米波雷达、摄像头、超声波雷达等。激光雷达提供精确的3D环境感知；摄像头负责识别标志、车道线等视觉信息；毫米波雷达在恶劣天气下表现更佳；超声波雷达主要用于近距离感知。"
        }
    ]
    
    print("添加知识记忆...")
    for knowledge in common_knowledge:
        content = MemoryContent(
            content=knowledge["content"],
            mime_type=MemoryMimeType.TEXT,
            metadata={
                "topic": knowledge["topic"],
                "type": "knowledge"
            }
        )
        await default_agent.agent_memory.add_to_short_term_memory(content, cancellation_token)
        print(f"✓ 添加知识: {knowledge['topic']}")
    
    # 关闭记忆系统，触发持久化
    await default_agent.agent_memory.close()
    
    print(f"✓ 默认初始记忆已创建并保存到: {DEFAULT_MEMORY_PATH}")


async def test_agent_with_default_memory():
    """测试使用默认记忆的智能体"""
    if not memory_exists(DEFAULT_MEMORY_PATH):
        print("❌ 默认记忆不存在，请先运行创建默认记忆的测试")
        return
    
    print("\n========== 测试使用默认记忆的智能体 ==========")
    
    # 创建记忆配置，使用默认记忆
    memory_config = BaseAgentMemoryConfig(
        name="default_memory",
        base_path=DEFAULT_MEMORY_PATH
    )
    
    # 创建智能体
    agent = BaseMemoryAgent(
        name="agent_with_default_memory",
        device_manager=manager,
        description="使用默认记忆的智能体",
        device_type="virtual",
        memory_config=memory_config
    )
    
    cancellation_token = CancellationToken()
    
    # 准备测试查询
    test_queries = [
        "自动驾驶汽车有哪些等级？",
        "电动汽车的电池类型有哪些？",
        "中国的充电基础设施如何？"
    ]
    
    print("\n测试对记忆中已有知识的查询:")
    for query in test_queries:
        print(f"\n查询: {query}")
        
        # 创建查询消息
        query_message = TextMessage(content=query, source="user")
        
        # 查询记忆
        memory_result = await agent.agent_memory.query(query, cancellation_token)
        print(f"找到相关记忆: {len(memory_result.results)} 条")
        
        # 实际调用智能体生成回答
        response = await agent.on_messages([query_message], cancellation_token)
        
        # 打印回答
        print(f"智能体回答: {response.chat_message.content[:200]}..." if len(response.chat_message.content) > 200 else f"智能体回答: {response.chat_message.content}")
    
    # 关闭记忆系统
    await agent.agent_memory.close()
    
    print("\n✓ 使用默认记忆的智能体测试完成")


async def main():
    """主测试函数"""
    print("===== 开始测试BaseMemoryAgent和BaseAgentMemory功能 =====\n")
    
    # 检查是否已经有持久化的记忆
    default_memory_exists = memory_exists(DEFAULT_MEMORY_PATH)
    
    # 根据命令行参数和记忆存在状态决定运行哪些测试
    # if args.all or not default_memory_exists:
    #     # 如果指定了运行所有测试，或者默认记忆不存在
    #     await test_memory_functionality()
    #     await test_memory_persistence()
    #     await test_memory_query_comparison()
        
    # 创建默认记忆（如果指定了force或者不存在）
    if args.force or not default_memory_exists:
        await create_default_memory()
    
    # 测试使用默认记忆的智能体
    await test_agent_with_default_memory()
    
    print("\n===== 所有测试完成 =====")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试BaseMemoryAgent和BaseAgentMemory功能")
    parser.add_argument("--all", default=False, action="store_true", help="运行所有测试，包括基础功能测试")
    parser.add_argument("--force", default=False, action="store_true", help="强制创建默认记忆，即使已存在")
    args = parser.parse_args()
    
    asyncio.run(main())
