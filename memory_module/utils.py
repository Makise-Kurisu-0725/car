import os
import pickle
import json
import requests
import numpy as np
import faiss
from typing import Any, Dict, List, Literal, Optional, Union

# 添加OpenAI客户端
try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    print("警告: 缺少openai库，请使用 'pip install openai' 安装")


class LLMClient:
    """
    大语言模型API客户端，提供模型调用和反思生成功能。
    
    该类用于与大型语言模型API进行交互，支持同步和异步两种方式生成反思内容。
    主要功能包括初始化客户端、异步生成反思、同步生成反思以及格式化反思消息。
    
    使用示例：
    
    构造函数参数：
    - api_key: API密钥，如果未提供则使用环境变量中的API_KEY。
    - base_url: API基础URL，默认为 硅基流动 "https://api.siliconflow.cn/v1"。
    - model: 使用的模型名称，默认为"Qwen/Qwen2.5-7B-Instruct"。
    
    特殊使用限制或潜在的副作用：
    - 如果未提供API密钥且环境变量中也没有API_KEY，则会使用默认备用密钥，并打印警告信息。
    - 如果未安装openai库，客户端初始化会失败，并打印警告信息。
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 base_url: str = "https://api.siliconflow.cn/v1",
                 model: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥，如果未提供则使用环境变量
            base_url: API基础URL
            model: 使用的模型名称
        """
        # 确保API密钥值可用
        if api_key is None:
            api_key = os.getenv("API_KEY")
            if api_key is None:
                api_key = 'sk-xxxxxx'  # 默认备用密钥
                print("警告: 未提供API密钥，使用默认备用密钥")
        
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        
        # 初始化客户端
        try:
            self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except NameError:
            self.async_client = None
            self.client = None
            print("警告: openai库未安装，客户端初始化失败")
    
    async def generate_reflection_async(self, 
                                  task_description: str, 
                                  message_history: List[Dict],
                                  temperature: float = 0.7,
                                  max_tokens: int = 1024) -> str:
        """
        异步生成关于任务的反思
        
        Args:
            task_description: 任务描述
            message_history: 对话历史消息列表
            temperature: 生成文本的随机性，较高的值会使输出更加随机
            max_tokens: 生成文本的最大标记数量
            
        Returns:
            生成的反思内容
        """
        if not self.async_client:
            raise RuntimeError("OpenAI客户端未正确初始化，请安装openai库")
        
        # 构建提示
        formatted_messages = self._format_messages_for_reflection(task_description, message_history)
        
        try:
            # 调用API
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 提取生成的文本
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                return "无法生成有效反思。"
        except Exception as e:
            print(f"生成反思时出错: {e}")
            return f"生成反思时发生错误: {str(e)}"
    
    def generate_reflection(self, 
                            task_description: str, 
                            message_history: List[Dict],
                            temperature: float = 0.7,
                            max_tokens: int = 1024) -> str:
        """
        同步生成关于任务的反思
        
        Args:
            task_description: 任务描述
            message_history: 对话历史消息列表
            temperature: 生成文本的随机性，较高的值会使输出更加随机
            max_tokens: 生成文本的最大标记数量
            
        Returns:
            生成的反思内容
        """
        if not self.client:
            raise RuntimeError("OpenAI客户端未正确初始化，请安装openai库")
        
        # 构建提示
        formatted_messages = self._format_messages_for_reflection(task_description, message_history)
        
        try:
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 提取生成的文本
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                return "无法生成有效反思。"
        except Exception as e:
            print(f"生成反思时出错: {e}")
            return f"生成反思时发生错误: {str(e)}"
    
    def _format_messages_for_reflection(self, task_description: str, message_history: List[Dict]) -> List[Dict]:
        """
        为反思任务格式化消息
        
        Args:
            task_description: 任务描述
            message_history: 对话历史
            
        Returns:
            格式化后的消息列表
        """
        # 创建系统消息，引导模型生成高质量的反思
        system_message = {
            "role": "system",
            "content": (
                "你是一个专注于任务反思和分析的助手。请基于以下对话历史，对指定任务进行深入反思。"
                "你的反思应该包含以下几个方面：\n"
                "1. 任务完成情况分析\n"
                "2. 用户主要关注点和兴趣\n"
                "3. 交互中的优点和不足\n"
                "4. 对未来类似任务的建议\n"
                "保持反思简洁但有深度，使用清晰的结构。"
            )
        }
        
        # 创建用户消息，包含任务描述和对话历史
        formatted_history = "\n\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
            for msg in message_history
        ])
        
        user_message = {
            "role": "user",
            "content": (
                f"请对以下任务进行反思分析：\n\n"
                f"任务: {task_description}\n\n"
                f"对话历史:\n{formatted_history}\n\n"
                f"请生成一个结构化的反思，评估任务完成情况，分析用户需求，并提出改进建议。"
            )
        }
        
        # 返回格式化的消息列表
        return [system_message, user_message]


class EmbeddingModelAPI:
    """
    嵌入模型API，用于将文本转换为向量。
    
    该类提供了与嵌入模型API进行交互的功能，主要包含初始化方法和获取文本嵌入向量的方法。
    初始化方法支持设置API的基础URL和API密钥，如果未提供API密钥，则会尝试从环境变量中获取。
    如果环境变量中也未找到API密钥，则会使用默认备用API密钥。
    
    核心功能：
    - 初始化API连接
    - 获取文本的嵌入向量
    
    使用示例：
    
    构造函数参数：
    - base_url: 硅基流动 API的基础URL，默认为"https://api.siliconflow.cn/v1/embeddings"。
    - api_key: API密钥，如果未提供则使用环境变量中的API密钥。
    
    特殊使用限制或潜在的副作用：
    - 如果未提供API密钥且环境变量中也未找到API密钥，则会使用默认备用API密钥，并打印警告信息。
    - 解析API响应时，如果遇到意外的响应格式，会打印警告信息并返回空列表。
    """
    
    def __init__(self,
                 base_url="https://api.siliconflow.cn/v1/embeddings", 
                 api_key=None):
        """
        初始化EmbeddingModelAPI类
        
        Args:
            base_url: API的基础URL
            api_key: API密钥，如果未提供则使用环境变量
        """
        self.url = base_url
        
        # 如果未提供API密钥，尝试从环境变量获取
        if api_key is None:
            api_key = os.getenv("API_KEY")
            if api_key is None:
                api_key = 'sk-xxxxxxx'  # 默认备用API密钥
                print("警告: 未提供API密钥，使用默认备用密钥")
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_embedding(self, model: str, input: List[str], encoding_format: str = "float") -> List[Dict[str, Any]]:
        """
        获取文本的嵌入向量
        
        Args:
            model: 使用的嵌入模型名称
            input: 输入文本列表
            encoding_format: 编码格式，默认为float
            
        Returns:
            包含嵌入向量的响应数据列表
        """
        payload = {
            "model": model,
            "input": input,
            "encoding_format": encoding_format
        }
        response = requests.request("POST", self.url, json=payload, headers=self.headers)
        return self._parse_response(response)
    
    def _parse_response(self, response) -> List[Dict[str, Any]]:
        """
        解析API响应
        
        Args:
            response: API响应对象
            
        Returns:
            解析后的数据列表
        """
        try:
            response_text = response.text
            response_json = json.loads(response_text)
            if isinstance(response_json, dict) and "data" in response_json:
                return response_json.get("data")
            elif isinstance(response_json, str):
                # 如果响应是字符串格式，尝试再次解析
                try:
                    data = json.loads(response_json)
                    if isinstance(data, dict) and "data" in data:
                        return data.get("data")
                except:
                    pass
            
            # 如果无法正确解析，打印出错误信息便于调试
            print(f"警告: 意外的响应格式: {response_text[:200]}...")
            
            # 返回空列表，避免报错
            return []
        except Exception as e:
            print(f"解析响应时出错: {e}")
            print(f"响应: {response.text[:200]}...")
            return []


class VectorDB:
    """
    基于FAISS的向量数据库类，支持文本的存储、检索、删除和持久化。
    该类使用指定的嵌入模型将文本转换为向量，并使用FAISS索引进行高效的相似度计算。
    支持多种相似度度量方式：
    - 'l2': L2欧氏距离 (越小越相似)
    - 'cosine': 余弦相似度 (越大越相似)
    - 'ip': 内积相似度 (越大越相似)
    
    使用示例：
    
    Args:
        path: 可选，从本地加载向量数据库的路径
        dimension: 向量维度，默认为512
        embedding_model: 使用的嵌入模型名称
        metric: 相似度度量方式，'l2'为欧氏距离，'cosine'为余弦相似度，'ip'为内积
        api_key: API密钥，如果未提供则使用环境变量
    """
    def __init__(self, 
                 path: Optional[str] = None, 
                 dimension: int = 512,
                 embedding_model: str = "BAAI/bge-m3",
                 metric: Literal['l2', 'cosine', 'ip'] = 'l2',
                 api_key: Optional[str] = None):
        """
        初始化向量数据库
        
        Args:
            path: 可选，从本地加载向量数据库的路径
            dimension: 向量维度，默认为512
            embedding_model: 使用的嵌入模型名称
            metric: 相似度度量方式，'l2'为欧氏距离，'cosine'为余弦相似度，'ip'为内积
            api_key: API密钥，如果未提供则使用环境变量
        """
        self.dimension = dimension
        self.embedding_model = embedding_model
        self.metric = metric
        self.normalize_vectors = (metric == 'cosine')  # 余弦相似度需要归一化向量
        
        # 确保API密钥值可用
        if api_key is None:
            api_key = os.getenv("API_KEY")
            if api_key is None:
                api_key = 'sk-xxxxxx'  # 默认API密钥，只用作备用
                print("警告: 未提供API密钥，使用默认备用密钥")
        
        # 初始化嵌入模型API
        self.embedding_api = EmbeddingModelAPI(api_key=api_key)
        
        # 初始化或加载向量索引和元数据
        if path and os.path.exists(f"{path}.meta"):
            self.load(path)
            
            # 重新设置嵌入API，确保加载后能正常使用
            self.embedding_api = EmbeddingModelAPI(api_key=api_key)
        else:
            # 创建FAISS索引，根据metric选择不同的索引类型
            self._create_index()
            
            # 存储文本及其对应的索引位置
            self.texts = []
            self.ids = []  # 用于跟踪和删除
            self.next_id = 0  # 用于生成唯一ID
            self.id_to_index = {}

    def _create_index(self):
        """根据选择的度量方式创建FAISS索引"""
        if self.metric == 'l2':
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.metric == 'cosine':
            # 余弦相似度使用内积，但需要向量归一化
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == 'ip':
            # 内积搜索 (Maximum Inner Product Search)
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"不支持的度量方式: {self.metric}")

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        归一化向量（用于余弦相似度）
        
        Args:
            vectors: 输入向量
            
        Returns:
            归一化后的向量
        """
        if not self.normalize_vectors:
            return vectors
            
        # 计算每个向量的L2范数
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # 防止除零
        norms = np.maximum(norms, 1e-8)
        # 归一化向量
        return vectors / norms

    def _get_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        获取文本的嵌入向量
        
        Args:
            text: 输入文本或文本列表
            
        Returns:
            嵌入向量或向量列表
        """
        if isinstance(text, str):
            text = [text]
            
        # 调用嵌入API获取向量
        response = self.embedding_api.get_embedding(self.embedding_model, text)
        
        # 提取向量
        embeddings = []
        for item in response:
            embeddings.append(np.array(item['embedding'], dtype=np.float32))
            
        vectors = np.array(embeddings)
        
        # 如果使用余弦相似度，需要归一化向量
        return self._normalize(vectors)

    def add(self, texts: Union[str, List[str]]) -> List[int]:
        """
        将文本添加到向量数据库
        
        Args:
            texts: 要添加的文本或文本列表
            
        Returns:
            添加的文本对应的ID列表
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # 获取嵌入向量（已经归一化，如果需要的话）
        embeddings = self._get_embedding(texts)
        
        # 添加到FAISS索引
        self.index.add(embeddings)
        
        # 记录文本和ID
        added_ids = []
        for i, text in enumerate(texts):
            current_id = self.next_id
            current_index = len(self.texts)
            
            self.texts.append(text)
            self.ids.append(current_id)
            self.id_to_index[current_id] = current_index
            
            added_ids.append(current_id)
            self.next_id += 1
            
        return added_ids

    def search(self, query: Union[str, List[str]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        在向量数据库中搜索与查询最相似的文本
        
        Args:
            query: 查询文本或文本列表
            top_k: 返回的最相似结果数量
            
        Returns:
            包含相似文本、相似度分数和ID的结果列表
        """
        if len(self.texts) == 0:
            return []
            
        if isinstance(query, str):
            query = [query]
            
        # 获取查询的嵌入向量（已经归一化，如果需要的话）
        query_embeddings = self._get_embedding(query)
        
        # 执行搜索
        scores, indices = self.index.search(query_embeddings, min(top_k, len(self.texts)))
        
        # 整理结果
        results = []
        for i, (score_array, idx_array) in enumerate(zip(scores, indices)):
            query_results = []
            for score, idx in zip(score_array, idx_array):
                if idx < 0 or idx >= len(self.texts):  # FAISS可能返回-1表示找不到结果
                    continue
                
                # 根据度量方式调整分数的解释
                if self.metric in ['cosine', 'ip']:
                    # 对于内积和余弦相似度，得分越高越好，不需要处理
                    similarity_score = float(score)
                else:  # l2
                    # 对于L2距离，得分越低越好，但为了统一接口，转换为相似度（越高越好）
                    # 使用一个简单的变换：相似度 = 1 / (1 + 距离)
                    similarity_score = float(1.0 / (1.0 + score))
                    
                query_results.append({
                    "text": self.texts[idx],
                    "score": similarity_score,
                    "raw_score": float(score),  # 保留原始分数
                    "id": self.ids[idx],
                    "metric": self.metric  # 添加度量方式说明
                })
            results.append(query_results)
            
        # 如果只有一个查询，就直接返回它的结果
        return results[0] if len(query) == 1 else results

    def delete(self, ids: Union[int, List[int]]) -> bool:
        """
        从向量数据库中删除指定ID的文本
        
        Args:
            ids: 要删除的文本ID或ID列表
            
        Returns:
            删除操作是否成功
        """
        if isinstance(ids, int):
            ids = [ids]
            
        # 检查FAISS索引是否支持删除
        if not isinstance(self.index, faiss.IndexIDMap):
            # 如果不支持，我们需要重建索引
            return self._rebuild_after_delete(ids)
            
        try:
            # 尝试直接删除
            self.index.remove_ids(np.array(ids, dtype=np.int64))
            
            # 更新元数据
            for id_to_remove in ids:
                if id_to_remove in self.id_to_index:
                    idx = self.id_to_index[id_to_remove]
                    self.texts[idx] = None  # 标记为已删除
                    del self.id_to_index[id_to_remove]
            
            return True
        except Exception as e:
            print(f"删除失败: {e}")
            # 回退到重建
            return self._rebuild_after_delete(ids)

    def _rebuild_after_delete(self, ids_to_remove: List[int]) -> bool:
        """
        通过重建索引的方式实现删除功能
        
        Args:
            ids_to_remove: 要删除的ID列表
            
        Returns:
            重建操作是否成功
        """
        try:
            # 筛选需要保留的内容
            keep_indices = [i for i, id_val in enumerate(self.ids) if id_val not in ids_to_remove]
            
            if not keep_indices:
                # 如果所有内容都被删除，重置数据库
                self._create_index()
                self.texts = []
                self.ids = []
                self.id_to_index = {}
                return True
                
            # 保留的文本
            kept_texts = [self.texts[i] for i in keep_indices]
            kept_ids = [self.ids[i] for i in keep_indices]
            
            # 获取这些文本的嵌入（已经归一化，如果需要的话）
            embeddings = self._get_embedding(kept_texts)
            
            # 重建索引
            self._create_index()
            self.index.add(embeddings)
            
            # 更新元数据
            self.texts = kept_texts
            self.ids = kept_ids
            self.id_to_index = {id_val: i for i, id_val in enumerate(kept_ids)}
            
            return True
        except Exception as e:
            print(f"重建索引失败: {e}")
            return False

    def save(self, path: str) -> bool:
        """
        将向量数据库保存到本地
        
        Args:
            path: 保存路径
            
        Returns:
            保存操作是否成功
        """
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # 准备保存的数据
            data_to_save = {
                "dimension": self.dimension,
                "embedding_model": self.embedding_model,
                "metric": self.metric,
                "normalize_vectors": self.normalize_vectors,
                "texts": self.texts,
                "ids": self.ids,
                "next_id": self.next_id,
                "id_to_index": self.id_to_index
            }
            
            # 保存元数据
            with open(f"{path}.meta", "wb") as f:
                pickle.dump(data_to_save, f)
                
            # 保存FAISS索引
            faiss.write_index(self.index, f"{path}.index")
            
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        从本地加载向量数据库
        
        Args:
            path: 加载路径
            
        Returns:
            加载操作是否成功
        """
        try:
            # 加载元数据
            with open(f"{path}.meta", "rb") as f:
                data = pickle.load(f)
                
            # 设置元数据
            self.dimension = data["dimension"]
            self.embedding_model = data["embedding_model"]
            self.metric = data.get("metric", "l2")  # 兼容旧版本
            self.normalize_vectors = data.get("normalize_vectors", self.metric == 'cosine')
            self.texts = data["texts"]
            self.ids = data["ids"]
            self.next_id = data["next_id"]
            self.id_to_index = data["id_to_index"]
            
            # 加载FAISS索引
            self.index = faiss.read_index(f"{path}.index")
            
            return True
        except Exception as e:
            print(f"加载失败: {e}")
            # 初始化一个空数据库
            self._create_index()
            self.texts = []
            self.ids = []
            self.next_id = 0
            self.id_to_index = {}
            return False
