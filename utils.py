from typing import Literal,Any
from autogen_agentchat.messages import BaseChatMessage
import json
import os
from typing import List, Dict, Optional
from device import  DeviceManager 
from autogen_core.tools import BaseTool
from openai import AsyncOpenAI

class ErrorMessage(BaseChatMessage):
    """A error message."""

    content: str
    """The content of the error message."""

    type: Literal["ErrorMessage"] = "ErrorMessage"



class AgentActionValidator:
    """智能代理动作验证器（使用OpenAI官方SDK）"""
    
    def __init__(
        self, 
        device_manager: DeviceManager,  # 替换为实际类型
        tools: List[BaseTool[Any, Any]],
        model_name: str = "Qwen/Qwen2.5-7B-Instruct"  # 推荐最新模型
    ):
        """
        :param device_manager: 设备管理器实例
        :param tools: 可用工具列表
        :param model_name: 模型名称
        """
        self.device_manager = device_manager
        self.tools = tools
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=os.getenv("API_KEY"),base_url="https://api.siliconflow.cn/v1")

    async def validate_execution(
        self,
        original_query: str,
        execution_result: str,
        device_type: Optional[type] = None
    ) -> Dict[str, Any]:
        """异步验证流程"""
        context = await self._build_validation_context(original_query, device_type)
        prompt = self._generate_validation_prompt(context, execution_result)
        
        try:
            response = await self._call_openai(prompt)
            return self._parse_response(response)
        except Exception as e:
            print(f"验证失败: {str(e)}")
            return {"is_valid": False, "reason": "验证服务不可用"}

    async def _call_openai(self, prompt: str):
        """使用官方SDK异步调用"""
        return await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
           response_format={"type": "json_object"}
        )

    def _parse_response(self, response) -> Dict:
        """解析OpenAI响应对象"""
        try:
            content = response.choices[0].message.content
            return json.loads(content)
        except (AttributeError, json.JSONDecodeError) as e:
            raise ValueError(f"响应解析失败: {str(e)}")

    # 以下方法保持原有逻辑不变 ----------
    async def _build_validation_context(
        self,
        original_query: str,
        device_type: Optional[type]
    ) -> Dict:
        return {
            "environment": self.device_manager.collect_environment_info(
                format_type='natural_language',
                device_type=device_type
            ),
            "tools": self._format_tools_info(),
            "original_query": original_query
        }

    def _format_tools_info(self) -> str:
        return "\n".join(
            f"- {tool.name}: {tool.description} (参数: {json.dumps(tool._args_type)})"
            for tool in self.tools
        )

    def _generate_validation_prompt(
        self, 
        context: Dict, 
        execution_result: str
    ) -> str:
        return f"""
        [系统角色] 你是一个智能系统验证器，负责检查操作是否完整满足用户需求,并返回指定的json。
        
        [设备状态]
        {context['environment']}
        
        [可用工具]
        {context['tools']}
        
        [用户请求]
        {context['original_query']}
        
        [执行结果]
        {execution_result}
        
        [验证要求]
        1. 如果[执行结果]是自然语言的回复，请你根据[设备状态]检查其回复是否与设备状态一致。
        2. 如果[执行结果]是包含工具执行结果和调用链的message，请确认设备状态变化是否满足了用户的请求
        
        [输出格式]
        {{
            "is_valid": boolean,
            "reason": "验证通过/失败的原因说明（中文）",
            "missing_requirements": ["未满足的具体需求项"]
        }}
        """

    async def close(self):
        """清理资源"""
        await self.client.close()