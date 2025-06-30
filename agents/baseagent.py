import json
import os
from typing import Sequence, List, AsyncGenerator

from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import FunctionExecutionResult, CreateResult,  AssistantMessage, UserMessage,FunctionExecutionResultMessage
from autogen_agentchat.base import Response
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import AgentEvent, TextMessage, ChatMessage,  ToolCallRequestEvent, ToolCallExecutionEvent, ThoughtEvent, ToolCallSummaryMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from utils import *



class BaseAgent(AssistantAgent):
    """基智能体""" 
    """添加工具调用错误下的重试机制、functioncall、记忆管理、向数据中心请求数据"""
    def __init__(self,name,description,device_type,check_mode:bool=True,max_retries=3, max_tool_retries=2, **kwargs):
        if name=="Li":
            model_client=OpenAIChatCompletionClient(
                model='Qwen/Qwen2.5-7B-Instruct',
                base_url="https://api.siliconflow.cn/v1",
                api_key=os.getenv("API_KEY"),
                response_format={"type": "json_object"},
                model_info={
                    "vision": False,
                    "function_calling": True,
                    "json_output": True,
                    "family": 'R1',
                },
            )
        else:
            model_client = OpenAIChatCompletionClient(
                model='Qwen/Qwen2.5-7B-Instruct',
                base_url="https://api.siliconflow.cn/v1",
                api_key=os.getenv("API_KEY"),
                model_info={
                    "vision": False,
                    "function_calling": True,
                    "json_output": True,
                    "family": 'Qwen2.5',
                }
            )
        self.device_type = device_type
        self._check_mode = check_mode
        super().__init__(name=name,model_client=model_client,description=description,**kwargs)
        self.max_retries = max_retries
        self.max_tool_retries = max_tool_retries

        self.validator = AgentActionValidator(
            device_manager=self.device_manager,
            tools=self._tools
        )
        

    
    
    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        """处理消息，增加环境信息并在推理后删除"""
        #messages = self._add_environment_info(messages)
        response = await super().on_messages(messages, cancellation_token)
        self._clean_environment_messages()
        print(f"\nagent {self.name}的执行结果为：{response}")

        #直接产生回复
        if isinstance(response.chat_message,TextMessage):
            if self.name!="Li" :
                if await self._check_agentcall_success(messages[0].content,response.chat_message):
                    response.chat_message.metadata["label"] = "success"
                else:
                    response.chat_message.metadata["label"] = "failed"
            return response
        
        #functioncall生成失败
        if isinstance(response.chat_message,ErrorMessage):
            response.chat_message.metadata["label"] = "pass"
            return response
        
        if isinstance(response.chat_message,ToolCallSummaryMessage):
            if await self._check_agentcall_success(messages[0].content,str(response)):
                response.chat_message.metadata["label"] = "success"
                return response
            else:
            #functioncall执行结果错误
                response.chat_message.metadata["label"] = "failed"
                return response


    async def _check_agentcall_success(self,query:str,agent_response:str)->bool:             
        """调用LLM，检查边缘agent产生的自然语言回复或functioncall的结果，是否完成了query。"""

        if self._check_mode:
            validation_result = await self.validator.validate_execution(
                original_query=query,
                execution_result=agent_response
            )
            assert validation_result["reason"]!="验证服务不可用",print("验证器调用出错")
            print("\nvalidator检查结果:",validation_result)
            return validation_result["is_valid"]
        else:
            return True
       

    def _add_environment_info(self, messages: Sequence[ChatMessage]) -> List[ChatMessage]:
        """为消息添加环境信息"""
        devices_info = self.device_manager.collect_environment_info(format_type='natural_language',device_type=self.device_type)
        print(devices_info)
        return [TextMessage(content=f"""<enviroment>
        {devices_info}
        </enviroment>""", source='user')] + list(messages)
        
    def _clean_environment_messages(self):
        """清理环境相关的消息"""
        for i in range(len(self._model_context._messages)-1, -1, -1):
            if self._model_context._messages[i].type == 'UserMessage' and self._model_context._messages[i].source == 'Li':
                self._model_context._messages.pop(i)
            if self._model_context._messages[i].type == 'UserMessage' and self._model_context._messages[i].content.startswith('<enviroment>'):
                self._model_context._messages.pop(i)
                break

    async def on_messages_stream(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:
        """重写流式处理方法，增加工具调用错误后的重试机制"""

        # 获取所有相关变量
        agent_name = self.name
        model_context = self._model_context
        memory = self._memory
        system_messages = self._system_messages
        tools = self._tools
        handoff_tools = self._handoff_tools
        handoffs = self._handoffs
        model_client = self._model_client
        model_client_stream = self._model_client_stream
        reflect_on_tool_use = self._reflect_on_tool_use
        tool_call_summary_format = self._tool_call_summary_format

        await self._add_messages_to_context(
            model_context=model_context,
            messages=messages,
        )

        inner_messages = []
        # for event_msg in await self._update_model_context_with_memory(
        #     memory=memory,
        #     model_context=model_context,
        #     agent_name=agent_name,
        # ):
        #     inner_messages.append(event_msg)
        #     yield event_msg

        retry_count = 0
        tool_execution_failed = False
        
        while retry_count <= self.max_retries:
            model_result = None
            async for inference_output in self._call_llm(
                model_client=model_client,
                model_client_stream=model_client_stream,
                system_messages=system_messages,
                model_context=model_context,
                tools=tools,
                handoff_tools=handoff_tools,
                agent_name=agent_name,
                cancellation_token=cancellation_token,
            ):
                if isinstance(inference_output, CreateResult):
                    model_result = inference_output
                else:
                    yield inference_output
    
            if model_result is None and retry_count < self.max_retries:
                    retry_count += 1
                    print(f"模型调用失败，重新发起调用：(第{retry_count}次)")
                    continue

            if model_result.thought:
                thought_event = ThoughtEvent(content=model_result.thought, source=agent_name)
                yield thought_event
                inner_messages.append(thought_event)

            await model_context.add_message(
                AssistantMessage(
                    content=model_result.content,
                    source=agent_name,
                    thought=getattr(model_result, "thought", None),
                )
            )
            # 验证模型结果是否为直接回复
            if isinstance(model_result.content, str):
                yield Response(
                    chat_message=TextMessage(
                        content=model_result.content,
                        source=agent_name,
                        models_usage=model_result.usage
                    ),
                    inner_messages=inner_messages,
                )

            try:
                # 验证模型结果是否为工具调用列表
                if not self._is_valid_function_call_list(model_result.content):
                    if retry_count < self.max_retries:
                        retry_count += 1
                        await self._add_retry_message(model_context, "请重新生成有效的工具调用，确保返回格式正确的FunctionCall列表。")
                        print(f"工具调用不合法。重试生成工具调用 (第{retry_count}次)")
                        continue
                    else:
                        yield Response(
                            chat_message=ErrorMessage(content="超过最大重试次数，无法生成有效的工具调用", source=self.name)
                        )
                
                # 处理模型输出并检查工具执行结果
                tool_execution_failed = False
                async for output_event in self._process_model_result_with_retry(
                    model_result=model_result,
                    inner_messages=inner_messages,
                    cancellation_token=cancellation_token,
                    agent_name=agent_name,
                    system_messages=system_messages,
                    model_context=model_context,
                    tools=tools,
                    handoff_tools=handoff_tools,
                    handoffs=handoffs,
                    model_client=model_client,
                    model_client_stream=model_client_stream,
                    reflect_on_tool_use=reflect_on_tool_use,
                    tool_call_summary_format=tool_call_summary_format,
                ):
                    # 检查是否有工具执行失败的事件
                    if isinstance(output_event, ToolCallExecutionEvent):
                        tool_execution_failed = self._check_tool_execution_failed(output_event)
                    
                    yield output_event
                    
                    if isinstance(output_event, Response):
                        self._clean_environment_messages()
                        yield Response(
                            chat_message=TextMessage(content="success", source=self.name)
                        )
                
                # 如果工具执行失败但仍有重试机会
                if tool_execution_failed and retry_count < self.max_retries:
                    retry_count += 1
                    await self._add_retry_message(model_context, "工具调用执行失败，请重新生成更准确的工具调用。")
                    print(f"工具调用执行失败。重试生成工具调用 (第{retry_count}次)")
                    continue
                
                # 工具执行成功或已达到最大重试次数，退出循环
                break
                
            except Exception as e:
                print(f"处理模型结果时出错: {str(e)}")
                if retry_count < self.max_retries:
                    retry_count += 1
                    await self._add_retry_message(model_context, f"处理错误: {str(e)}。请重新生成有效的工具调用。")
                    print(f"工具调用执行失败。重试生成工具调用 (第{retry_count}次)")
                    continue
                else:
                    # 已达到最大重试次数，返回错误响应
                    yield Response(
                        chat_message=ErrorMessage(
                            content=f"多次尝试后，工具调用执行仍然出错: {str(e)}",
                            source=agent_name
                        ),
                        inner_messages=inner_messages,
                    )
                    return

    def _is_valid_function_call_list(self, content):
        """检查内容是否为有效的函数调用列表"""
        return isinstance(content, list) and all(
            isinstance(item, FunctionCall) for item in content
        )
        
    def _check_tool_execution_failed(self, output_event):
        """检查工具执行是否失败"""
        for result in output_event.content:
            if result.is_error:
                print(f"工具调用失败: {result.content}")
                return True
        return False
        
    async def _add_retry_message(self, model_context, message):
        """添加重试消息到模型上下文"""
        await model_context.add_message(
            UserMessage(
                content=message,
                source="system"
            )
        )

    async def _process_model_result_with_retry(
        self,
        model_result,
        inner_messages,
        cancellation_token,
        agent_name,
        system_messages,
        model_context,
        tools,
        handoff_tools,
        handoffs,
        model_client,
        model_client_stream,
        reflect_on_tool_use,
        tool_call_summary_format,
    ):
        """处理模型结果并执行工具调用，支持在出错时重试"""
        
        # 验证工具调用
        assert self._is_valid_function_call_list(model_result.content)

        # 产生工具调用请求事件
        tool_call_msg = ToolCallRequestEvent(
            content=model_result.content,
            source=agent_name,
            models_usage=model_result.usage,
        )
        inner_messages.append(tool_call_msg)
        yield tool_call_msg

        # 执行工具调用，支持在出错时重试
        executed_calls_and_results = []
        for call in model_result.content:
            result = await self._execute_tool_call_with_retry(
                call, 
                tools, 
                handoff_tools, 
                agent_name, 
                cancellation_token, 
                model_context
            )
            executed_calls_and_results.append(result)

        # 处理执行结果
        exec_results = [result for _, result in executed_calls_and_results]
        tool_call_result_msg = ToolCallExecutionEvent(
            content=exec_results,
            source=agent_name,
        )
        await model_context.add_message(FunctionExecutionResultMessage(content=exec_results))
        inner_messages.append(tool_call_result_msg)
        yield tool_call_result_msg

        # 检查是否需要处理handoff
        handoff_output = self._check_and_handle_handoff(
            model_result=model_result,
            executed_calls_and_results=executed_calls_and_results,
            inner_messages=inner_messages,
            handoffs=handoffs,
            agent_name=agent_name,
        )
        if handoff_output:
            yield handoff_output
            return

        # 处理工具使用反思或总结
        if reflect_on_tool_use:
            async for reflection_response in self._reflect_on_tool_use_flow(
                system_messages=system_messages,
                model_client=model_client,
                model_client_stream=model_client_stream,
                model_context=model_context,
                agent_name=agent_name,
                inner_messages=inner_messages,
            ):
                yield reflection_response
        else:
            yield self._summarize_tool_use(
                executed_calls_and_results=executed_calls_and_results,
                inner_messages=inner_messages,
                handoffs=handoffs,
                tool_call_summary_format=tool_call_summary_format,
                agent_name=agent_name,
            )
    
    async def _execute_tool_call_with_retry(self, call, tools, handoff_tools, agent_name, cancellation_token, model_context):
        """执行工具调用并处理重试逻辑"""
        success = False
        
        for attempt in range(self.max_tool_retries + 1):
            try:
                # 检查和修复参数格式
                fixed_arguments = self._normalize_function_arguments(call.arguments, call.name)
                
                # 创建修复后的调用对象
                fixed_call = FunctionCall(
                    id=call.id, 
                    name=call.name, 
                    arguments=fixed_arguments
                )
                
                print(f"尝试执行工具调用 {call.name}，参数: {fixed_arguments}")
                
                result = await self._execute_tool_call(
                    tool_call=fixed_call,
                    tools=tools,
                    handoff_tools=handoff_tools,
                    agent_name=agent_name,
                    cancellation_token=cancellation_token,
                )
                
                return result
                
            except Exception as e:
                error_message = str(e)
                print(f"工具调用 {call.name} 失败 (尝试 {attempt+1}/{self.max_tool_retries+1}): {error_message}")
                
                # 特别处理参数验证错误
                if "validation error" in error_message and attempt < self.max_tool_retries:
                    print("检测到参数验证错误，尝试修复参数格式...")
                    await self._add_retry_message(
                        model_context, 
                        f"工具 {call.name} 调用失败，参数验证错误: {error_message}。请重新生成正确格式的参数。"
                    )
                    continue
        
        # 所有重试都失败
        return (
            call,
            FunctionExecutionResult(
                content=f"工具调用失败: {error_message}",
                call_id=call.id,
                is_error=True,
                name=call.name,
            ),
        )
            
    def _normalize_function_arguments(self, arguments, function_name):
        """规范化函数参数，处理格式问题"""
        try:
            # 检查参数是否已经是字符串形式的JSON
            if isinstance(arguments, str):
                # 处理双重引用的JSON字符串
                if arguments.startswith('"') and arguments.endswith('"'):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        # 处理转义字符
                        import re
                        unquoted = arguments[1:-1]
                        unescaped = re.sub(r'\\(.)', r'\1', unquoted)
                        arguments = unescaped
                
                # 尝试解析JSON字符串
                try:
                    parsed_args = json.loads(arguments)
                    return json.dumps(parsed_args)  # 确保返回标准格式的JSON字符串
                except json.JSONDecodeError:
                    print(f"警告: 参数不是有效的JSON字符串: {arguments}")
                    return arguments
            
            # 对字典类型进行序列化
            elif isinstance(arguments, dict):
                return json.dumps(arguments)
            
            # 返回原始参数
            return arguments
            
        except Exception as e:
            print(f"参数处理过程中发生错误: {str(e)}")
            return arguments    

    async def __aexit__(self, *args):
        """确保资源释放"""
        await self.validator.close()
        await super().__aexit__(*args)

