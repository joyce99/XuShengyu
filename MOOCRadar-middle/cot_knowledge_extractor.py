# -*- coding: utf-8 -*-
"""
基于CoT推理路径的隐式知识点提取器
Reasoning Path Analysis + Potential Knowledge Mapping & Extraction

逻辑流程:
1. 推理路径解析: 使用CoT Prompt让LLM展示逐步推理过程，生成多条解题思路
2. 知识点抽取: 从每条思路的每个步骤中抽取候选知识点(Premise)
3. 潜在知识点映射: 将候选知识点与知识库进行向量相似度匹配
4. 阈值筛选: 筛选出相似度超过阈值的知识点作为隐式关联知识点
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from zhipuai import ZhipuAI
from openai import OpenAI
import faiss
import pickle
from pathlib import Path
import time
import random
import re
import config  # 导入配置文件，加载环境变量


class CoTKnowledgeExtractor:
    def __init__(self,
                 concept_mapping_file: str = "data/concept_mapping.json",
                 index_dir: str = "data/vector_index",
                 qwen_api_key: str = None,
                 embedding_api_key: str = None,
                 embedding_base_url: str = None,
                 llm_model: str = "qwen-max",
                 similarity_threshold: float = 0.6,
                 sleep_range: Tuple[float, float] = (0.5, 1.5),
                 enable_thinking: bool = False):
        """
        初始化CoT知识点提取器
        
        Args:
            concept_mapping_file: 知识点映射文件路径 (KC全集)
            index_dir: 向量索引保存目录
            qwen_api_key: 阿里云千问API密钥
            embedding_api_key: 嵌入模型API密钥
            embedding_base_url: 嵌入模型API地址
            llm_model: LLM模型名称 (qwen-plus, qwq-32b等)
            similarity_threshold: 相似度阈值 τ2
            sleep_range: 请求之间的随机延迟范围(秒)
            enable_thinking: 是否启用深度思考模式
        """
        # 初始化千问客户端 - 阿里云千问 API
        self.qwen_client = OpenAI(
            api_key=qwen_api_key or os.environ.get("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 初始化本地模型客户端 - 本地部署的模型
        self.local_client = OpenAI(
            api_key="None",
            base_url="http://127.0.0.1:12345/v1",
        )
        
        # 根据模型名称选择客户端
        if llm_model and ('qwen3' in llm_model or 'deepseek' in llm_model):
            self.client = self.local_client  # 本地模型
        else:
            self.client = self.qwen_client  # 云端千问
        
        self.enable_thinking = enable_thinking
        
        # 初始化OpenAI客户端 - 嵌入模型使用
        self.embedding_client = OpenAI(
            api_key=embedding_api_key or os.environ.get("EMBEDDING_API_KEY"),
            base_url=embedding_base_url or os.environ.get("EMBEDDING_BASE_URL"),
        )
        
        self.llm_model = llm_model
        self.similarity_threshold = similarity_threshold
        self.sleep_range = sleep_range
        
        # 加载知识点映射 (KC全集)
        self._load_concept_mapping(concept_mapping_file)
        
        # 初始化向量索引
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self._init_vector_index()
        
        print(f"初始化完成:")
        print(f"  - 知识点数量: {len(self.concept_mapping)}")
        print(f"  - LLM模型: {self.llm_model}")
        print(f"  - 相似度阈值 τ2: {self.similarity_threshold}")

    def _load_concept_mapping(self, file_path: str) -> None:
        """加载知识点映射文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.concept_mapping = json.load(f)
            print(f"成功加载知识点映射: {file_path}")
        except Exception as e:
            print(f"加载知识点映射失败: {str(e)}")
            self.concept_mapping = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的向量表示"""
        try:
            response = self.embedding_client.embeddings.create(
                model="text-embedding-bge-m3",
                input=text
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"获取向量失败: {str(e)}")
            return np.array([])

    def _init_vector_index(self):
        """初始化或加载FAISS索引"""
        index_path = self.index_dir / "faiss_index.bin"
        vectors_path = self.index_dir / "topic_vectors.pkl"
        
        if index_path.exists() and vectors_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                with open(vectors_path, 'rb') as f:
                    data = pickle.load(f)
                    self.topic_vectors = data['vectors']
                    self.topic_ids = data['ids']
                print("成功加载现有向量索引")
                return
            except Exception as e:
                print(f"加载索引失败: {str(e)}")
        
        print("构建新的向量索引...")
        topic_vectors = []
        topic_ids = []
        
        for topic_id, topic_name in tqdm(self.concept_mapping.items(), desc="构建知识点向量"):
            topic_desc = f"知识点：{topic_name}"
            vector = self._get_embedding(topic_desc)
            if len(vector) > 0:
                topic_vectors.append(vector)
                topic_ids.append(topic_id)
        
        self.topic_vectors = np.array(topic_vectors, dtype=np.float32)
        self.topic_ids = np.array(topic_ids)
        
        dimension = len(self.topic_vectors[0])
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.topic_vectors)
        self.index.add(self.topic_vectors)
        
        # 保存索引
        faiss.write_index(self.index, str(index_path))
        with open(vectors_path, 'wb') as f:
            pickle.dump({'vectors': self.topic_vectors, 'ids': self.topic_ids}, f)
        print("向量索引构建完成")

    def _random_delay(self) -> None:
        """随机延迟"""
        delay = random.uniform(self.sleep_range[0], self.sleep_range[1])
        time.sleep(delay)

    def _robust_json_parse(self, json_str: str, original_content: str) -> Dict[str, Any]:
        """
        强健的JSON解析，尝试多种修复方式
        """
        # 方法1: 直接解析
        try:
            return json.loads(json_str)
        except:
            pass
        
        # 方法2: 修复非法转义字符
        try:
            # 移除所有非法转义 (保留 " \ / b f n r t u)
            fixed = re.sub(r'\\(?!["\\/bfnrtu])', '', json_str)
            return json.loads(fixed)
        except:
            pass
        
        # 方法3: 更激进的修复 - 替换所有反斜杠
        try:
            fixed = json_str.replace('\\', '')
            return json.loads(fixed)
        except:
            pass
        
        # 方法4: 修复单引号问题
        try:
            # 将属性名的单引号替换为双引号
            fixed = re.sub(r"'(\w+)':", r'"\1":', json_str)
            fixed = re.sub(r'\\(?!["\\/bfnrtu])', '', fixed)
            return json.loads(fixed)
        except:
            pass
        
        # 方法5: 尝试提取premise_set数组
        try:
            match = re.search(r'"premise_set"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
            if match:
                premises_str = match.group(1)
                # 提取所有引号中的内容
                premises = re.findall(r'"([^"]+)"', premises_str)
                return {"reasoning_paths": [], "premise_set": premises}
        except:
            pass
        
        # 方法6: 从原始文本中提取premise
        premise_set = self._extract_premise_from_text(original_content)
        if premise_set:
            return {"reasoning_paths": [], "premise_set": premise_set}
        
        # 最后还是失败，抛出异常
        raise json.JSONDecodeError("无法解析JSON", json_str, 0)

    def _extract_premise_from_text(self, text: str) -> List[str]:
        """
        从文本中提取知识点（当JSON解析失败时的备选方案）
        """
        premises = set()
        
        # 尝试匹配 "premise": "xxx" 模式
        matches = re.findall(r'"premise"\s*:\s*"([^"]+)"', text)
        premises.update(matches)
        
        # 尝试匹配 premise_set 数组中的内容
        match = re.search(r'"premise_set"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if match:
            array_content = match.group(1)
            items = re.findall(r'"([^"]+)"', array_content)
            premises.update(items)
        
        # 尝试匹配 知识点: xxx 模式
        matches = re.findall(r'知识点[：:]\s*([^\n,，]+)', text)
        for m in matches:
            m = m.strip().strip('"\'')
            if m and len(m) < 50:  # 过滤太长的内容
                premises.add(m)
        
        return list(premises)

    def reasoning_path_analysis(self, exercise_text: str) -> Dict[str, Any]:
        """
        推理路径解析 (Reasoning Path Analysis)
        
        使用CoT Prompt让LLM展示逐步推理过程，生成多条解题思路，
        并从每个步骤中抽取候选知识点(Premise)
        
        P_ej = LLM(e_text_j | CoT-Prompt)
        
        Args:
            exercise_text: 习题文本
            
        Returns:
            Dict: {
                "reasoning_paths": [
                    {
                        "path_id": 1,
                        "path_name": "思路名称",
                        "steps": [
                            {"step_id": 1, "description": "步骤描述", "premise": "涉及的知识点"}
                        ]
                    }
                ],
                "premise_set": ["知识点1", "知识点2", ...]  # 所有Premise的集合
            }
        """
        cot_prompt = """你作为该习题学科领域的专家，请对习题文本进行结构化链式思维推理（CoT）。要求：

1. 生成多条可能的解题思路（至少2条，如果有多种解法）
2. 每条思路拆分为按序编号的步骤（Step 1, Step 2, ...），并写清楚推理过程
3. 在每一步标注使用到的知识点（Premise）
4. 最后汇总所有出现过的知识点的Premise集合

请严格按照以下JSON格式输出：
{
    "reasoning_paths": [
        {
            "path_id": 1,
            "path_name": "思路名称（如：代数展开法）",
            "steps": [
                {
                    "step_id": 1,
                    "description": "具体的推理步骤描述",
                    "premise": "该步骤涉及的知识点名称"
                }
            ]
        }
    ],
    "premise_set": ["知识点1", "知识点2", "知识点3"]
}

注意：
- 每个步骤的premise应该是具体的知识点名称，如"平方定义"、"乘法分配律"、"合并同类项"等
- premise_set是所有步骤中涉及的知识点的去重集合
- 只输出JSON，不要输出其他内容"""

        user_message = f"""请对以下习题进行结构化链式思维推理：

{exercise_text}"""

        try:
            # 使用千问深度思考模型，支持流式输出
            if self.enable_thinking:
                completion = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": cot_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    extra_body={"enable_thinking": True},
                    stream=True
                )
                
                # 收集思考过程和回复内容
                thinking_content = ""
                answer_content = ""
                is_answering = False
                
                print("\n" + "=" * 20 + "思考过程" + "=" * 20)
                for chunk in completion:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                        if not is_answering:
                            print(delta.reasoning_content, end="", flush=True)
                            thinking_content += delta.reasoning_content
                    if hasattr(delta, "content") and delta.content:
                        if not is_answering:
                            print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                            is_answering = True
                        print(delta.content, end="", flush=True)
                        answer_content += delta.content
                
                print("\n" + "=" * 50)
                content = answer_content
            else:
                # 不启用深度思考，普通调用
                # 根据模型选择客户端
                if self.llm_model.startswith('qwen'):
                    # 千问模型使用阿里云 API
                    response = self.client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": cot_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        temperature=0.3,
                        max_tokens=4000
                    )
                else:
                    # 其他模型（本地部署）使用 OpenAI 兼容接口
                    response = self.client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": cot_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        temperature=0.3,
                        max_tokens=4000
                    )
                content = response.choices[0].message.content
                print("=" * 50)
                print("CoT推理结果:")
                print(content[:500] + "..." if len(content) > 500 else content)
                print("=" * 50)
            
            if not content or not content.strip():
                raise ValueError('LLM返回内容为空')
            
            # 解析JSON
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # 提取JSON对象
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                json_str = content
            
            # 尝试多种方式解析JSON
            result = self._robust_json_parse(json_str, content)
            
            # 确保premise_set存在
            if "premise_set" not in result:
                # 从steps中提取所有premise
                premises = set()
                for path in result.get("reasoning_paths", []):
                    for step in path.get("steps", []):
                        if step.get("premise"):
                            premises.add(step["premise"])
                result["premise_set"] = list(premises)
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            # 尝试从原始内容中提取premise
            premise_set = self._extract_premise_from_text(content if 'content' in dir() else "")
            return {
                "reasoning_paths": [],
                "premise_set": premise_set,
                "error": f"JSON解析失败: {str(e)}" if not premise_set else None
            }
        except Exception as e:
            print(f"推理路径分析出错: {str(e)}")
            return {
                "reasoning_paths": [],
                "premise_set": [],
                "error": str(e)
            }

    def potential_knowledge_mapping(self, 
                                    premise_set: List[str], 
                                    top_k: int = 5) -> Dict[str, Any]:
        """
        潜在知识点映射与提取 (Potential Knowledge Mapping & Extraction)
        
        对LLM推理生成的候选知识点集合与已知的知识点集合进行相似度判断，
        抽取出隐性关联知识点。
        
        KC_imp(e_j) = {c_k} | sim_imp(p^{t,l}_{e_j}, c_k) > τ2, c_k ∈ KC
        
        Args:
            premise_set: CoT推理得到的候选知识点集合 P_{e_j}
            top_k: 每个premise匹配的最大知识点数量
            
        Returns:
            Dict: {
                "premise_mappings": {
                    "premise_name": [
                        {"knowledge_id": "id", "knowledge_name": "name", "similarity": 0.8}
                    ]
                },
                "implicit_knowledge": [
                    {"knowledge_id": "id", "knowledge_name": "name", "similarity": 0.8, "from_premise": "xxx"}
                ]
            }
        """
        premise_mappings = {}
        all_matched_knowledge = {}  # 用于去重，key为knowledge_id
        
        for premise in premise_set:
            if not premise:
                continue
                
            # 获取premise的向量表示
            premise_vector = self._get_embedding(f"知识点：{premise}")
            if len(premise_vector) == 0:
                continue
            
            # 归一化
            faiss.normalize_L2(premise_vector.reshape(1, -1))
            
            # 搜索最相似的知识点
            similarities, indices = self.index.search(
                premise_vector.reshape(1, -1), 
                top_k
            )
            
            matched = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:
                    continue
                    
                similarity = float(similarities[0][i])
                
                # 只保留超过阈值的
                if similarity >= self.similarity_threshold:
                    knowledge_id = str(self.topic_ids[idx])
                    knowledge_name = self.concept_mapping.get(knowledge_id, "未知")
                    
                    match_info = {
                        "knowledge_id": knowledge_id,
                        "knowledge_name": knowledge_name,
                        "similarity": round(similarity, 4)
                    }
                    matched.append(match_info)
                    
                    # 更新全局匹配（保留最高相似度）
                    if knowledge_id not in all_matched_knowledge or \
                       all_matched_knowledge[knowledge_id]["similarity"] < similarity:
                        all_matched_knowledge[knowledge_id] = {
                            "knowledge_id": knowledge_id,
                            "knowledge_name": knowledge_name,
                            "similarity": round(similarity, 4),
                            "from_premise": premise
                        }
            
            premise_mappings[premise] = matched
            
            print(f"Premise '{premise}' 匹配到 {len(matched)} 个知识点")
        
        # 整理隐式知识点列表（按相似度排序）
        implicit_knowledge = sorted(
            all_matched_knowledge.values(),
            key=lambda x: x["similarity"],
            reverse=True
        )
        
        return {
            "premise_mappings": premise_mappings,
            "implicit_knowledge": implicit_knowledge,
            "implicit_knowledge_ids": [k["knowledge_id"] for k in implicit_knowledge],
            "implicit_knowledge_names": [k["knowledge_name"] for k in implicit_knowledge]
        }

    def extract_implicit_knowledge(self, exercise_text: str) -> Dict[str, Any]:
        """
        完整的隐式知识点提取流程
        
        1. 推理路径解析: P_ej = LLM(e_text_j | CoT-Prompt)
        2. 潜在知识点映射: KC_imp(e_j) = {c_k} | sim(p, c_k) > τ2
        
        Args:
            exercise_text: 习题文本
            
        Returns:
            Dict: 完整的提取结果
        """
        result = {
            "exercise_text": exercise_text,
            "reasoning_paths": [],
            "premise_set": [],
            "premise_mappings": {},
            "implicit_knowledge": [],
            "implicit_knowledge_ids": [],
            "implicit_knowledge_names": []
        }
        
        # Step 1: 推理路径解析
        print("\n[Step 1] 推理路径解析 (Reasoning Path Analysis)...")
        cot_result = self.reasoning_path_analysis(exercise_text)
        
        result["reasoning_paths"] = cot_result.get("reasoning_paths", [])
        result["premise_set"] = cot_result.get("premise_set", [])
        
        if cot_result.get("error"):
            result["error"] = cot_result["error"]
            return result
        
        print(f"生成 {len(result['reasoning_paths'])} 条推理路径")
        print(f"提取 {len(result['premise_set'])} 个候选知识点(Premise)")
        
        self._random_delay()
        
        # Step 2: 潜在知识点映射
        print("\n[Step 2] 潜在知识点映射 (Potential Knowledge Mapping)...")
        mapping_result = self.potential_knowledge_mapping(result["premise_set"])
        
        result["premise_mappings"] = mapping_result["premise_mappings"]
        result["implicit_knowledge"] = mapping_result["implicit_knowledge"]
        result["implicit_knowledge_ids"] = mapping_result["implicit_knowledge_ids"]
        result["implicit_knowledge_names"] = mapping_result["implicit_knowledge_names"]
        
        print(f"映射得到 {len(result['implicit_knowledge'])} 个隐式关联知识点")
        
        return result

    def process_from_json(self,
                          json_file: str = "problem_formatted.json",
                          id_mapping_file: str = "problem_id_mapping.json",
                          output_file: str = None,
                          start_index: int = 0,
                          max_count: int = None) -> pd.DataFrame:
        """
        从problem_formatted.json批量处理习题，按problem_id_mapping.json顺序
        
        Args:
            json_file: 习题JSON文件路径
            id_mapping_file: 问题ID映射文件路径（定义处理顺序）
            output_file: 输出文件路径
            start_index: 起始索引
            max_count: 最大处理数量
            
        Returns:
            pd.DataFrame: 处理结果
        """
        if output_file is None:
            threshold_str = str(self.similarity_threshold).replace('.', '_')
            if start_index > 0:
                output_file = f"cot_implicit_knowledge_t{threshold_str}_from{start_index}.xlsx"
            else:
                output_file = f"cot_implicit_knowledge_t{threshold_str}.xlsx"
        
        try:
            # 加载习题数据
            with open(json_file, 'r', encoding='utf-8') as f:
                exercises = json.load(f)
            
            # 建立 problem_id -> exercise 的映射
            exercise_map = {str(ex.get('problem_id')): ex for ex in exercises}
            print(f"成功读取习题文件: {json_file}, 共 {len(exercises)} 条习题")
            
            # 加载问题ID映射（定义处理顺序）
            if os.path.exists(id_mapping_file):
                with open(id_mapping_file, 'r', encoding='utf-8') as f:
                    id_mapping = json.load(f)
                print(f"成功加载ID映射文件: {id_mapping_file}, 共 {len(id_mapping)} 条映射")
                
                # 按映射顺序排列 problem_id 列表
                ordered_problem_ids = [id_mapping[str(i)] for i in range(len(id_mapping))]
            else:
                print(f"未找到ID映射文件 {id_mapping_file}，按原始顺序处理")
                ordered_problem_ids = [str(ex.get('problem_id')) for ex in exercises]
            
            total_rows = len(ordered_problem_ids)
            end_index = total_rows if max_count is None else min(start_index + max_count, total_rows)
            process_count = end_index - start_index
            print(f"将处理索引 {start_index} 到 {end_index-1}, 共 {process_count} 条")
            
            results = []
            
            for i in tqdm(range(start_index, end_index), desc="处理进度"):
                problem_id = ordered_problem_ids[i]
                
                # 根据 problem_id 获取习题
                if problem_id not in exercise_map:
                    print(f"警告: 索引 {i} 的 problem_id '{problem_id}' 在习题文件中不存在，跳过")
                    continue
                
                exercise = exercise_map[problem_id]
                
                # 获取习题内容
                if not exercise.get('detail'):
                    print(f"警告: 索引 {i} 习题内容为空，跳过")
                    continue
                
                detail = exercise['detail'][0]
                
                # 处理 detail 是字符串的情况
                if isinstance(detail, str):
                    content = detail
                    options = {}
                else:
                    content = detail.get('content', '')
                    options = detail.get('option', {})
                
                if not content:
                    print(f"警告: 索引 {i} 习题内容为空，跳过")
                    continue
                
                # 组合习题文本
                if options is None or not options:
                    exercise_text = content
                else:
                    options_text = '\n'.join([f"{k}: {v}" for k, v in options.items()])
                    exercise_text = f"{content}\n\n选项：\n{options_text}"
                
                exercise_name = detail.get('title', f"习题_{problem_id}") if isinstance(detail, dict) else f"习题_{problem_id}"
                
                print(f"\n{'='*60}")
                print(f"处理第 {i+1}/{total_rows} 条: {exercise_name}")
                print(f"problem_id: {problem_id}")
                print(f"{'='*60}")
                
                try:
                    # 提取隐式知识点
                    result = self.extract_implicit_knowledge(exercise_text)
                    
                    # 添加元信息
                    result["row_index"] = i
                    result["problem_id"] = problem_id
                    result["exercise_name"] = exercise_name
                    
                    # 简化reasoning_paths用于存储
                    result["reasoning_paths_json"] = json.dumps(
                        result["reasoning_paths"], ensure_ascii=False
                    )
                    result["premise_mappings_json"] = json.dumps(
                        result["premise_mappings"], ensure_ascii=False
                    )
                    result["implicit_knowledge_json"] = json.dumps(
                        result["implicit_knowledge"], ensure_ascii=False
                    )
                    
                    # 移除复杂对象，保留可序列化字段
                    result.pop("reasoning_paths", None)
                    result.pop("premise_mappings", None)
                    result.pop("implicit_knowledge", None)
                    
                    results.append(result)
                    
                    print(f"\n结果汇总:")
                    print(f"  - Premise集合: {result['premise_set']}")
                    print(f"  - 隐式知识点ID: {result['implicit_knowledge_ids']}")
                    print(f"  - 隐式知识点名称: {result['implicit_knowledge_names']}")
                    
                    # 每10条保存一次
                    if (i - start_index + 1) % 10 == 0:
                        temp_df = pd.DataFrame(results)
                        temp_df.to_excel(output_file, index=False)
                        print(f"\n已保存 {len(results)} 条结果到: {output_file}")
                    
                    self._random_delay()
                    
                except Exception as e:
                    print(f"处理索引 {i} 时出错: {str(e)}")
                    results.append({
                        "row_index": i,
                        "problem_id": problem_id,
                        "exercise_name": exercise_name,
                        "exercise_text": exercise_text,
                        "premise_set": [],
                        "implicit_knowledge_ids": [],
                        "implicit_knowledge_names": [],
                        "error": str(e)
                    })
            
            # 保存最终结果
            result_df = pd.DataFrame(results)
            result_df.to_excel(output_file, index=False)
            print(f"\n处理完成, 共处理 {len(results)} 条记录")
            print(f"结果已保存到: {output_file}")
            
            return result_df
            
        except Exception as e:
            print(f"处理文件时出错: {str(e)}")
            return pd.DataFrame()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='基于CoT推理路径的隐式知识点提取器（千问深度思考版）'
    )
    parser.add_argument('--file', type=str, 
                        default="data/problem_formatted.json",
                        help='习题JSON文件路径')
    parser.add_argument('--mapping', type=str,
                        default="data/problem_id_mapping.json",
                        help='问题ID映射文件路径（定义处理顺序）')
    parser.add_argument('--output', type=str, 
                        default=None,
                        help='输出文件路径')
    parser.add_argument('--start', type=int, 
                        default=0,
                        help='起始索引')
    parser.add_argument('--count', type=int, 
                        default=None,
                        help='处理数量')
    parser.add_argument('--threshold', type=float, 
                        default=0.6,
                        help='相似度阈值 τ2 (默认: 0.6)')
    parser.add_argument('--model', type=str,
                        default="qwen-max",
                        help='LLM模型名称 (qwen-max, qwen-plus, qwq-32b等)')
    parser.add_argument('--no-thinking', action='store_true',
                        help='禁用深度思考模式')
    
    args = parser.parse_args()
    
    extractor = CoTKnowledgeExtractor(
        similarity_threshold=args.threshold,
        llm_model=args.model,
        enable_thinking=not args.no_thinking
    )
    
    extractor.process_from_json(
        json_file=args.file,
        id_mapping_file=args.mapping,
        output_file=args.output,
        start_index=args.start,
        max_count=args.count
    )


if __name__ == "__main__":
    main()
