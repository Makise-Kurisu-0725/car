import os
import json
import asyncio
import dotenv
from fastapi import FastAPI, WebSocket,WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from agents.agents import LiAgent, SeatAgent, MeituanAgent, AVAgent, NavigationAgent, UseAgent, TextMessage, CancellationToken
import uvicorn
from threading import Thread, Event
from agents.agents import *

# 加载环境变量
dotenv.load_dotenv()

# ========== 全局状态管理 ==========
class AppState:
    def __init__(self):
        self.manager = DeviceManager()
        self.li_agent = None
        self.server_ready = Event()
    
    def get_device_state(self):
        data = app_state.manager.collect_environment_info()
        return json.loads(data)

app_state = AppState()

# ========== 连接管理器 ==========
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

connection_manager = ConnectionManager()

# 继承main.py中的初始化逻辑但改造为服务模式
class CarService:
    def __init__(self,manager:DeviceManager):
        self.manager = manager
        self.li = self._init_agents()
        self.task_queue = asyncio.Queue()
        self.current_tasks: Dict[str, asyncio.Task] = {}  # 跟踪各用户任务

    def _init_agents(self):
        # 创建座椅Agent
        seat_agent = SeatAgent(
            self.manager, 
            name="seat_agent", 
            description="座椅智能体。可以获取与设置特定位置座椅的信息。包括座椅按摩模式，按摩档位、座椅加热、座椅通风模式、座椅占位信息。"
        )
        
        # 创建美团Agent
        meituan_agent = MeituanAgent(
            self.manager, 
            name="meituan_agent", 
            description="美团外卖订餐智能体。可以搜索不同类型的商家，查询菜单，点餐(将菜品加入购物车)，设置配送地点，以及完成支付。"
        )

        # 创建音视频Agent
        av_agent = AVAgent(
            self.manager, 
            name="audio_agent", 
            description="音视频智能体。可以控制车内音乐播放器和视频播放器。"
        )

        # 创建出行Agent
        navigation_agent = NavigationAgent(
            self.manager, 
            name="navigation_agent", 
            description="出行智能体。可以搜索导航路线。"
        )

        # 创建用车Agent
        use_agent = UseAgent(
            self.manager, 
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
            device_manager = self.manager
        )
        return Li

    async def process_query(self, user_id: str, query: str):
        """处理用户查询的核心方法"""
        try:
            # 1. 创建消息
            message = TextMessage(content=query, source=user_id)
            
            # 2. 获取执行计划
            plan = await self.li.on_messages(
                [message], 
                cancellation_token=CancellationToken()
            )
            
            if plan.chat_message.content[0] == '错':
                return {"status": "error", "message": plan.chat_message.content}
            
            from pprint import pprint
            pprint(json.loads(plan.chat_message.content))
            # 3. 执行计划
            execution = await self.li.execute_plan(
                plan.chat_message.content,
                CancellationToken()
            )

            if execution:
                return {
                    "status": "success",
                    "plan": json.loads(plan.chat_message.content)
                }
            else:
                return {
                    "status": "error",
                    "message": "任务执行错误"
                }

        except asyncio.TimeoutError:
            return {"status": "timeout", "message": "操作执行超时"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

class QueryRequest(BaseModel):
    query: str
    user_id: str


# ========== 应用生命周期 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # 初始化智能体系统
    #service.li = await service = CarService(app_state.manager)
    
    # 标记服务准备就绪
    app_state.server_ready.set()
    
    yield
    
    # 清理资源
    app_state.manager.cleanup()

service = CarService(app_state.manager)
app = FastAPI(lifespan=lifespan)

# ========== CORS配置 ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== API路由 ==========
@app.get("/api/environment-info")
async def get_environment_info():
    return app_state.get_device_state()

# ========== WebSocket路由 ==========
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)

# ========== 并行用户query处理路由 ==========
@app.post("/api/query")
async def handle_query(request: QueryRequest):
    """
    处理用户查询入口
    - user_id: 用于区分不同用户会话
    - query: 用户指令文本
    """
    # 将查询加入处理队列
    task = asyncio.create_task(
        service.process_query(request.user_id, request.query)
    )
    service.current_tasks[request.user_id] = task
    
    try:
        result = await task
        return result
    finally:
        service.current_tasks.pop(request.user_id, None)

        


# ========== 测试指令 ==========
async def execute_tasks():
    # 等待服务初始化完成
    while not app_state.server_ready.is_set():
        print("等待服务初始化...")
        await asyncio.sleep(0.5)
    
    query_list = [
        {"content": "请将左后门的乘客按摩强度调到最高，并将主驾的座椅通风打开", "agent": "座椅agent"},
        # 其他测试用例...
    ]

    for task in query_list:
        print(f"\n=== {task['agent']} 测试 ===")
        
        try:
            # 生成执行计划
            plan = await service.li.on_messages(
                [TextMessage(content=task["content"], source="user")],
                CancellationToken()
            )
            
            # 广播任务开始
            await connection_manager.broadcast(json.dumps({
                "type": "task_start",
                "task": task["content"],
                "data": app_state.get_device_state()
            }))

            # 执行计划
            await service.li.execute_plan(
                plan.chat_message.content, 
                CancellationToken()
            )

            # 广播任务完成
            await connection_manager.broadcast(json.dumps({
                "type": "task_complete",
                "task": task["content"],
                "data": app_state.get_device_state()
            }))
            
            print("任务执行成功")
            
        except Exception as e:
            error_msg = f"执行失败: {str(e)}"
            print(error_msg)
            await connection_manager.broadcast(json.dumps({
                "type": "error",
                "message": error_msg
            }))

# ========== 启动入口 ==========
if __name__ == "__main__":
    # 创建事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # 启动FastAPI服务
    server_config = uvicorn.Config(
        app, 
        host="0.0.0.0", 
        port=8000,
        loop="asyncio"
    )
    server = uvicorn.Server(server_config)
    
    # 定义服务器启动任务
    async def run_server():
        await server.serve()
    
    # 定义主任务
    async def main():
        # 并行运行服务器和任务执行
        await asyncio.gather(
            run_server(),
            execute_tasks()
        )
    
    # 运行主事件循环
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\n服务已终止")
    finally:
        loop.close()