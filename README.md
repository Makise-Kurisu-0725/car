# AutoCar
理想多智能体智能座舱模拟平台

## 本周TODO：3.21
实现Demo 1：
1. 完善BaseDevice、BaseAgent和DeviceManager的设计
2. Agent类统一加入agent.py。边缘Agent继承自BaseAgent类
3. 设备类统一加入device.py。实例设备继承自BaseDevice类
4. 边缘Agent向理想同学汇报执行结果：
执行完任务后，输入设备的状态，让Agent检查是否成功完成。并向理想同学智能体发起通信。
5. 引入多步functioncall：
目前AutoGen似乎天然支持多个function的生成和自动执行
6. 与理想的API接口对齐


## 本周进展：4.1
1. 各个agent、对应的device、functioncall接口的独立设计基本完成。
2. 基于zcg在functioncall阶段引入纠错的版本，设计了BaseAgent，重构所有agent的代码。

TODO：
1. 设计长短期记忆的管理机制-> zcg
2. 实现version1.0的主动感知模块 ->  负责边缘智能体的 each one
3. 优化理想同学的并发任务管理，以及主函数中并发的向每个智能体发起主动感知轮询。 -> zsw, zpf
4. 选做   实现一个简单的可视化界面 -> zsw