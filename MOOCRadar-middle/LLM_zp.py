import os
import json
import config
import pandas as pd
from typing import List, Dict, Union, Tuple
from tqdm import tqdm
from zhipuai import ZhipuAI
from openai import OpenAI
import numpy as np
import faiss
import pickle
from pathlib import Path
import time
import random

class GPTKnowledgePredictor:
    def __init__(self, topics_file: str = "concept_mapping.json", 
                 index_dir: str = "vector_index",
                 analysis_top_k: int = 30,
                 result_top_k: int = 3,
                 score_threshold: float = 0.6,
                 gpt_api_key: str = None,
                 gpt_base_url: str = None,
                 embedding_api_key: str = None,
                 embedding_base_url: str = None,
                 unlimited_threshold: bool = False,
                 llm_model: str = "glm-4-air-250414"  # 新增：支持指定模型
                 ):
        """
        初始化预测器
        Args:
            topics_file: 知识点JSON文件路径
            index_dir: 向量索引保存目录
            analysis_top_k: 分析阶段的候选知识点数量
            result_top_k: 最终返回的知识点数量
            score_threshold: 知识点选择的综合评分阈值
            gpt_api_key: GPT API密钥
            gpt_base_url: GPT API地址
            embedding_api_key: 嵌入模型API密钥
            embedding_base_url: 嵌入模型API地址
            unlimited_threshold: 是否使用无限制阈值模式
            llm_model: LLM模型名称（如 glm-4-air, deepseek-r1-distill-llama-8b 等）
        """
        if not os.path.exists(topics_file):
            raise FileNotFoundError(f"找不到知识点文件: {topics_file}")
            
        self.analysis_top_k = analysis_top_k
        self.result_top_k = result_top_k
        self.score_threshold = score_threshold
        self.llm_model = llm_model  # 保存模型名称
        
        # 标记是否使用自定义阈值模式（不限制返回数量）
        self._custom_threshold_set = unlimited_threshold
            
        # 初始化ZhipuAI客户端 - GLM模型使用
        self.zhipu_client = ZhipuAI(
            api_key=gpt_api_key or os.environ.get("ZHIPUAI_API_KEY"),
        )
        
        # 初始化阿里云千问客户端 - 云端千问模型
        self.qwen_client = OpenAI(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 初始化本地模型客户端 - 本地部署的模型
        self.local_client = OpenAI(
            api_key="None",
            base_url="http://127.0.0.1:12345/v1",
        )
        
        # 根据模型名称选择主客户端
        if llm_model.startswith('glm'):
            self.client = self.zhipu_client
        elif 'qwen3' in llm_model or 'deepseek' in llm_model:
            self.client = self.local_client
        elif llm_model.startswith('qwen'):
            self.client = self.qwen_client
        else:
            self.client = self.local_client  # 默认使用本地
        
        # 初始化嵌入模型客户端（本地部署的 BGE-M3）
        self.embedding_client = OpenAI(
            api_key="None",
            base_url="http://127.0.0.1:12345/v1",
        )
            
        # 读取JSON文件
        with open(topics_file, 'r', encoding='utf-8') as f:
            self.knowledge_points = json.load(f)
        
        # 初始化向量索引
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self.index_path = self.index_dir / "faiss_index.bin"
        self.vectors_path = self.index_dir / "topic_vectors.pkl"
        
        # 构建或加载向量索引
        self._init_vector_index()

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
        if self._load_existing_index():
            print("成功加载现有向量索引")
            return

        print("构建新的向量索引...")
        topic_vectors = []
        topic_ids = []
        
        for topic_id, topic_name in tqdm(self.knowledge_points.items(), desc="构建知识点向量"):
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

        self._save_index()

    def _load_existing_index(self) -> bool:
        """加载现有的索引文件"""
        try:
            if not (self.index_path.exists() and self.vectors_path.exists()):
                return False

            self.index = faiss.read_index(str(self.index_path))
            
            with open(self.vectors_path, 'rb') as f:
                data = pickle.load(f)
                self.topic_vectors = data['vectors']
                self.topic_ids = data['ids']
            return True
        except Exception as e:
            print(f"加载索引失败: {str(e)}")
            return False

    def _save_index(self):
        """保存索引和向量数据"""
        try:
            faiss.write_index(self.index, str(self.index_path))
            
            with open(self.vectors_path, 'wb') as f:
                pickle.dump({
                    'vectors': self.topic_vectors,
                    'ids': self.topic_ids
                }, f)
        except Exception as e:
            print(f"保存索引失败: {str(e)}")

    def _get_fallback_result(self, similarities, indices) -> Dict[str, Dict]:
        """生成备选的分析结果"""
        return {
            str(self.topic_ids[idx]): {
                "name": self.knowledge_points[self.topic_ids[idx]],
                "relevance": float(similarities[0][i]),
                "evidence": "基于向量相似度匹配",
                "explanation": "自动匹配结果"
            }
            for i, idx in enumerate(indices[0])
            if idx != -1
        }

    def _analyze_knowledge_aspects(self, exercise_text: str) -> Dict[str, Dict]:
        """
        分析练习文本涉及的知识点
        Returns:
            Dict[str, Dict]: 知识点分析结果
        """
        # 首先用embedding找出最相关的知识点
        query_vector = self._get_embedding(f"习题：{exercise_text}")
        faiss.normalize_L2(query_vector.reshape(1, -1))
        
        # 搜索最相似的前N个知识点（默认30个）
        similarities, indices = self.index.search(
            query_vector.reshape(1, -1), 
            self.analysis_top_k
        )
        
        # 构建候选知识点列表
        candidate_points = {}
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                topic_id = str(self.topic_ids[idx])
                candidate_points[topic_id] = {
                    "name": self.knowledge_points[self.topic_ids[idx]],
                    "similarity": float(similarities[0][i])
                }

        messages = [
            {
                "role": "system", 
                "content": """你是一个多学科教育专家。请分析这道习题涉及的知识点。

对于每个相关的知识点，请按照以下格式输出（不要输出其他任何内容）：

知识点 <知识点ID> (<知识点名称>):
相关度: <0到1之间的数值>
证据: <从习题中提取的关键文本>
解释: <为什么这个知识点相关的简要说明>

示例输出：
知识点 3163 (DNA的复制):
相关度: 0.94
证据: 习题中描述了DNA复制的过程和机制
解释: 习题内容明确涉及DNA复制的基本原理和过程

注意：
1. 相关度评分要客观，基于习题内容与知识点的实际关联程度
2. 证据要具体，引用习题中的相关文本
3. 解释要简明扼要，说明知识点与习题内容的关联
4. 只输出确实相关的知识点，不要过度关联"""
            },
            {
                "role": "user",
                "content": f"请分析以下习题涉及的知识点：\n\n{exercise_text}"
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model="glm-4-air-250414",
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            # 解析响应
            result = {}
            current_topic = None
            current_data = {}
            
            for line in response.choices[0].message.content.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('知识点'):
                    # 保存前一个知识点
                    if current_topic and current_data:
                        result[current_topic] = current_data
                    
                    # 开始新知识点
                    parts = line.split('(')
                    if len(parts) >= 2:
                        topic_id = parts[0].replace('知识点', '').strip()
                        current_topic = topic_id
                        current_data = {
                            'name': parts[1].rstrip('):'),
                            'relevance': 0.0,
                            'evidence': '',
                            'explanation': ''
                        }
                elif line.startswith('相关度:'):
                    try:
                        current_data['relevance'] = float(line.replace('相关度:', '').strip())
                    except:
                        current_data['relevance'] = 0.0
                elif line.startswith('证据:'):
                    current_data['evidence'] = line.replace('证据:', '').strip()
                elif line.startswith('解释:'):
                    current_data['explanation'] = line.replace('解释:', '').strip()
            
            # 保存最后一个知识点
            if current_topic and current_data:
                result[current_topic] = current_data
                
            return result
            
        except Exception as e:
            print(f"分析知识点时出错: {str(e)}")
            return self._get_fallback_result(similarities, indices)

    def _get_knowledge_based_embeddings(self, exercise_text: str) -> Dict[int, List[Dict]]:
        """获取基于知识点的向量表示和LLM分析结果"""
        try:
            # 1. 获取习题的向量表示
            exercise_vector = self._get_embedding(exercise_text)
            if len(exercise_vector) == 0:
                return {}

            # 2. 先用向量相似度筛选候选知识点
            candidate_knowledge = {}
            vector_similarities = {}
            
            for topic_id, topic_name in self.knowledge_points.items():
                topic_desc = f"知识点：{topic_name}"
                topic_vector = self._get_embedding(topic_desc)
                if len(topic_vector) > 0:
                    # 计算向量相似度
                    vector_similarity = float(np.dot(exercise_vector, topic_vector))
                    vector_similarities[topic_id] = vector_similarity
            
            # 按相似度排序，选取前N个作为候选
            top_n = min(self.analysis_top_k, len(vector_similarities))
            candidates = sorted(vector_similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            print(f"已筛选出 {len(candidates)} 个候选知识点进行深入分析")
            
            # 3. 只对候选知识点使用LLM分析
            candidate_ids = [c[0] for c in candidates]
            candidate_names = [self.knowledge_points[c_id] for c_id in candidate_ids]
            
            # 构建候选知识点描述
            candidate_desc = "\n".join([f"{i+1}. {name} (ID: {c_id})" 
                                       for i, (c_id, name) in enumerate(zip(candidate_ids, candidate_names))])
            
            # 使用LLM分析习题与候选知识点的相关度
            llm_results = self._analyze_knowledge_with_candidates(exercise_text, candidate_ids, candidate_desc)

            # 4. 整合向量相似度和LLM分析结果
            knowledge_embeddings = {}
            for topic_id in candidate_ids:
                # 获取向量相似度
                vector_similarity = vector_similarities[topic_id]
                
                # 获取LLM分析的相关度（如果有）
                llm_relevance = 0.0
                llm_evidence = ""
                if topic_id in llm_results:
                    llm_relevance = llm_results[topic_id].get('relevance', 0.0)
                    llm_evidence = llm_results[topic_id].get('evidence', "")
                
                knowledge_embeddings[topic_id] = [{
                    'llm_relevance': llm_relevance,  # LLM分析的相关度
                    'vector_similarity': vector_similarity,  # 向量相似度
                    'evidence': llm_evidence or f"基于'{self.knowledge_points[topic_id]}'的向量相似度"
                }]

            return knowledge_embeddings

        except Exception as e:
            print(f"获取知识点向量表示时出错: {str(e)}")
            return {}

    def _search_with_knowledge_embeddings(
        self, 
        embeddings: Dict[int, List[Dict]]
    ) -> List[Tuple[int, float, str]]:
        """使用基于知识点的向量进行相似度搜索"""
        try:
            results = []
            
            # 权重设置
            llm_weight = 0.3  # LLM相关度权重
            vector_weight = 0.7  # 向量相似度权重
            
            for topic_id, vectors_info in embeddings.items():
                for vec_info in vectors_info:
                    # 获取两个分数
                    llm_relevance = vec_info['llm_relevance']  # LLM相关度
                    vector_similarity = vec_info['vector_similarity']  # 向量相似度
                    evidence = vec_info['evidence']
                    
                    # 计算加权综合得分
                    combined_score = (llm_relevance * llm_weight) + (vector_similarity * vector_weight)
                    
                    # 只添加超过阈值的结果
                    if combined_score >= self.score_threshold:
                        results.append((topic_id, combined_score, evidence))
            
            # 根据 score_threshold 是否为默认值决定返回策略
            if hasattr(self, '_custom_threshold_set') and self._custom_threshold_set:
                # 如果设置了自定义阈值，则返回所有超过阈值的结果（按分数排序）
                return sorted(results, key=lambda x: x[1], reverse=True)
            else:
                # 否则按照原来的逻辑，返回前 result_top_k 个结果
                return sorted(results, key=lambda x: x[1], reverse=True)[:self.result_top_k]
            
        except Exception as e:
            print(f"搜索知识点时出错: {str(e)}")
            return []

    def predict_single(self, exercise_text: str, challenge_id: str = None) -> Dict:
        """
        预测单个习题的知识点
        Returns:
            Dict: {
                'success': bool,
                'knowledge_points': List[Dict],  # 预测的知识点详细信息
                'error': str  # 如果失败，错误信息
            }
        """
        try:
            # 1. 基于知识点分析获取向量表示
            knowledge_embeddings = self._get_knowledge_based_embeddings(exercise_text)
            if not knowledge_embeddings:
                return {
                    'success': False,
                    'knowledge_points': [],
                    'error': '无法获取知识点向量表示'
                }

            # 2. 使用知识点相关的向量进行搜索
            similar_topics = self._search_with_knowledge_embeddings(knowledge_embeddings)
            
            # 3. 构建详细的结果信息
            result_points = []
            print("\n知识点匹配结果：")
            
            # 在无限制阈值模式下显示知识点数量
            if hasattr(self, '_custom_threshold_set') and self._custom_threshold_set:
                print(f"无限制阈值模式: 找到 {len(similar_topics)} 个超过阈值 {self.score_threshold} 的知识点")
            
            # 权重设置
            llm_weight = 0.3  # LLM相关度权重
            vector_weight = 0.7  # 向量相似度权重
            
            for topic_id, combined_score, evidence in similar_topics:
                # 从knowledge_embeddings中获取相关度
                topic_info = next(iter(knowledge_embeddings[topic_id]))
                llm_relevance = topic_info['llm_relevance']  # LLM相关度
                vector_similarity = topic_info['vector_similarity']  # 向量相似度
                
                point_info = {
                    'topic_id': topic_id,
                    'name': self.knowledge_points[topic_id],
                    'llm_relevance': llm_relevance,  # LLM相关度
                    'vector_similarity': vector_similarity,  # 向量相似度
                    'combined_score': combined_score,  # 综合得分
                    'evidence': evidence  # 关键证据
                }
                result_points.append(point_info)
                
                print(f"\n知识点 {topic_id}: {self.knowledge_points[topic_id]}")
                print(f"LLM相关度 (权重{llm_weight}): {llm_relevance:.3f}")
                print(f"向量相似度 (权重{vector_weight}): {vector_similarity:.3f}")
                print(f"综合得分: {combined_score:.3f}")
                print(f"关键证据: {evidence}")
                if challenge_id:
                    print(f"习题ID: {challenge_id}")

            return {
                'success': True,
                'knowledge_points': result_points,
                'error': None
            }

        except Exception as e:
            error_msg = str(e)
            print(f"预测过程出错: {error_msg}")
            return {
                'success': False,
                'knowledge_points': [],
                'error': error_msg
            }

    def predict_batch(self, exercise_texts: List[str]) -> List[List[int]]:
        """批量预测习题的知识点"""
        results = []
        for text in tqdm(exercise_texts, desc="预测进度"):
            results.append(self.predict_single(text))
        return results

    def get_knowledge_names(self, knowledge_ids: List[int]) -> List[str]:
        """获取知识点ID对应的名称"""
        return [
            self.knowledge_points.get(kid, "未知知识点") 
            for kid in knowledge_ids
        ]

    def predict_from_json(
        self, 
        json_file: str, 
        output_file: str, 
        error_file: str,
        start_index: int = 0
    ) -> None:
        """从JSON文件批量预测知识点"""
        try:
            # 1. 加载数据
            with open(json_file, 'r', encoding='utf-8') as f:
                exercises = json.load(f)
            
            # 2. 加载问题ID映射
            mapping_file = 'problem_id_mapping.json'
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    id_mapping = json.load(f)
                print(f"加载了 {len(id_mapping)} 个问题ID映射")
            else:
                id_mapping = {}
                print("未找到问题ID映射文件")
            
            # 3. 数据验证
            print(f"\n数据验证:")
            print(f"- problem_formatted.json 中的记录数: {len(exercises)}")
            print(f"- problem_id_mapping.json 中的映射数: {len(id_mapping)}")
            
            if len(exercises) != len(id_mapping):
                print(f"\n警告: 两个文件的记录数不匹配!")
                print(f"- 建议检查数据文件是否完整")
                print(f"- 是否继续处理? (y/n)")
                response = input().strip().lower()
                if response != 'y':
                    print("已取消处理")
                    return
            
            # 4. 准备结果存储
            results = [None] * len(exercises)  # 预分配与习题数量相同的空位
            errors = []  # 存储所有错误
            
            # 判断是否为新创建的输出文件
            is_new_output_file = "_start" in output_file
            
            # 如果输出文件存在且不是新创建的文件，则加载已有结果
            if os.path.exists(output_file) and not is_new_output_file:
                try:
                    existing_df = pd.read_excel(output_file)
                    print(f"已加载已有结果文件，包含 {len(existing_df)} 行数据")
                    
                    # 将已有结果填充到对应位置
                    for _, row in existing_df.iterrows():
                        if 'problem_id' in row and row['problem_id'] is not None:
                            # 查找这个problem_id对应的序号
                            for idx, mapped_id in id_mapping.items():
                                if mapped_id == row['problem_id']:
                                    index = int(idx)
                                    # 只保存start_index之前的结果，start_index及之后的将被重新处理
                                    if 0 <= index < start_index and index < len(results):
                                        results[index] = row.to_dict()
                                        print(f"恢复已有结果: problem_id={row['problem_id']} 到位置 {index}")
                                    break
                except Exception as e:
                    print(f"加载已有结果失败: {e}")
                    # 不影响继续执行
            elif is_new_output_file:
                print(f"使用新输出文件: {output_file}")
                print(f"将只处理从索引 {start_index} 开始的习题，且不加载已有结果")
            
            # 5. 处理每个习题
            total = len(exercises)
            processed_count = 0
            batch_count = 0  # 当前批次计数
            
            # 如果使用新文件，且起始索引不为0，则清空新文件开始处的None值
            if is_new_output_file and start_index > 0:
                results = [None] * (len(exercises) - start_index)
            
            for i, exercise in enumerate(exercises):
                # 跳过start_index之前的所有记录
                if i < start_index:
                    continue
                
                try:
                    print(f"\n处理第 {i+1}/{total} 条记录:")
                    
                    # 获取习题ID和名称
                    problem_id = exercise.get('problem_id', f'unknown_id_{i}')
                    name = f"Problem {problem_id}"
                    
                    # 获取习题内容
                    if not exercise.get('detail'):
                        raise ValueError("习题内容为空")
                    
                    # 获取第一个detail的内容
                    detail = exercise['detail'][0]
                    content = detail.get('content', '')
                    if not content:
                        raise ValueError("习题内容为空")
                    
                    # 获取选项
                    options = detail.get('option', {})
                    
                    # 组合完整习题文本
                    if options is None or not options:
                        # 选项为null或空，只使用content
                        exercise_text = f"{content}"
                        print("选项为空，仅使用习题内容")
                    else:
                        # 选项不为空，组合内容和选项
                        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
                    exercise_text = f"{content}\n\n选项：\n{options_text}"
                    
                    print(f"problem_id: {problem_id}")
                    print(f"name: {name}")
                    
                    # 预测知识点
                    result = self.predict_single(exercise_text, problem_id)
                    
                    if result['success']:
                        # 获取知识点名称
                        knowledge_names = []
                        for point in result['knowledge_points']:
                            topic_id = point['topic_id']
                            if topic_id in self.knowledge_points:
                                knowledge_names.append(self.knowledge_points[topic_id])
                            else:
                                knowledge_names.append(f"未知知识点({topic_id})")
                        
                        # 构建结果行
                        result_row = {
                            'problem_id': problem_id,
                            'name': name,
                            'knowledge_count': len(result['knowledge_points']),
                            'knowledge_ids': [point['topic_id'] for point in result['knowledge_points']],
                            'knowledge_names': knowledge_names,
                            'llm_relevance_scores': [point['llm_relevance'] for point in result['knowledge_points']],
                            'vector_similarities': [point['vector_similarity'] for point in result['knowledge_points']],
                            'combined_scores': [point['combined_score'] for point in result['knowledge_points']],
                            'evidences': [point['evidence'] for point in result['knowledge_points']],
                            'error': None
                        }
                        
                                                # 将结果保存到适当位置
                        if is_new_output_file:
                            # 如果是新文件，保存到相对位置（从0开始）
                            results[i - start_index] = result_row
                        else:
                            # 如果是旧文件，保存到原始位置
                            results[i] = result_row
                        
                        processed_count += 1
                        batch_count += 1
                    else:
                        error_row = {
                            'problem_id': problem_id,
                            'name': name,
                            'error_type': '预测失败',
                            'error_message': result['error']
                        }
                        errors.append(error_row)
                        
                        # 记录空结果
                        empty_result = {
                            'problem_id': problem_id,
                            'name': name,
                            'knowledge_count': 0,
                            'knowledge_ids': [],
                            'knowledge_names': [],
                            'llm_relevance_scores': [],
                            'vector_similarities': [],
                            'combined_scores': [],
                            'evidences': [],
                            'error': result['error']
                        }
                        
                        # 保存到适当位置
                        if is_new_output_file:
                            results[i - start_index] = empty_result
                        else:
                            results[i] = empty_result
                        
                        batch_count += 1
                    
                    # 每处理10条记录保存一次
                    if batch_count >= 10:
                        # 过滤None值，只保存有效结果
                        valid_results = [r for r in results if r is not None]
                        print(f"\n当前有效结果数: {len(valid_results)}/{len(results)}")
                        
                        # 保存结果
                        result_df = pd.DataFrame(valid_results)
                        result_df.to_excel(output_file, index=False)
                        print(f"已保存 {len(valid_results)} 条结果记录到: {output_file}")
                        
                        # 保存错误日志
                        if errors:
                            error_df = pd.DataFrame(errors)
                            error_df.to_excel(error_file, index=False)
                            print(f"已保存 {len(errors)} 条错误记录到: {error_file}")
                        
                        print(f"\n已保存进度: {i+1}/{total}，本批次处理了 {batch_count} 条")
                        # 重置批次计数
                        batch_count = 0
                
                except Exception as e:
                    print(f"错误: {str(e)}")
                    error_row = {
                        'problem_id': problem_id if 'problem_id' in locals() else f'unknown_id_{i}',
                        'name': name if 'name' in locals() else f'Problem unknown_id_{i}',
                        'error_type': '处理异常',
                        'error_message': str(e)
                    }
                    errors.append(error_row)
                    
                    # 记录空结果
                    empty_result = {
                        'problem_id': problem_id if 'problem_id' in locals() else f'unknown_id_{i}',
                        'name': name if 'name' in locals() else f'Problem unknown_id_{i}',
                        'knowledge_count': 0,
                        'knowledge_ids': [],
                        'knowledge_names': [],
                        'llm_relevance_scores': [],
                        'vector_similarities': [],
                        'combined_scores': [],
                        'evidences': [],
                        'error': str(e)
                    }
                    
                    # 保存到适当位置
                    if is_new_output_file:
                        results[i - start_index] = empty_result
                    else:
                        results[i] = empty_result
                    
                    batch_count += 1
                    
                    # 如果这批次已经处理了10条，也保存一次
                    if batch_count >= 10:
                        valid_results = [r for r in results if r is not None]
                        result_df = pd.DataFrame(valid_results)
                        result_df.to_excel(output_file, index=False)
                        print(f"\n已保存进度: {i+1}/{total}，本批次处理了 {batch_count} 条")
                        batch_count = 0
            
            # 7. 保存最终结果
            # 过滤None值，只保存有效结果
            valid_results = [r for r in results if r is not None]
            print(f"\n最终有效结果数: {len(valid_results)}/{len(results)}")
            
            result_df = pd.DataFrame(valid_results)
            result_df.to_excel(output_file, index=False)
            print(f"已保存 {len(valid_results)} 条结果记录到: {output_file}")
            
            if errors:
                error_df = pd.DataFrame(errors)
                error_df.to_excel(error_file, index=False)
                print(f"已保存 {len(errors)} 条错误记录到: {error_file}")
                
            print(f"\n处理完成，共处理 {processed_count} 条记录")
            
        except Exception as e:
            print(f"处理JSON文件时出错: {str(e)}")
            raise

    def predict_from_excel(self, excel_file: str, name_col: str = 'name', 
                          content_col: str = 'summarized_content', 
                          start_index: int = 0,
                          output_file: str = None,
                          error_file: str = None) -> pd.DataFrame:
        """
        从Excel文件中读取习题，预测知识点并返回结果
        Args:
            excel_file: Excel文件路径
            name_col: 习题名称列名
            content_col: 习题内容列名
            start_index: 开始处理的索引，用于断点续传
            output_file: 输出结果文件路径，如果为None则自动生成
            error_file: 错误日志文件路径，如果为None则自动生成
        Returns:
            pd.DataFrame: 预测结果DataFrame
        """
        try:
            # 读取输入文件
            input_df = pd.read_excel(excel_file)
            input_row_count = len(input_df)
            print(f"成功读取文件: {excel_file}, 总行数: {input_row_count}")
            
            # 确定输出文件名
            if output_file is None:
                # 根据是否使用自定义阈值决定文件名前缀
                mode = "unlimited" if hasattr(self, '_custom_threshold_set') and self._custom_threshold_set else "limited"
                # 文件名中添加阈值
                threshold_str = str(self.score_threshold).replace('.', '_')
                # 文件名中添加起始行号
                start_part = f"_from{start_index}" if start_index > 0 else ""
                
                output_file = f'prediction_results_{mode}_threshold{threshold_str}{start_part}_zp.xlsx'
            
            if error_file is None:
                # 错误文件也添加起始行号
                error_part = f"_from{start_index}" if start_index > 0 else ""
                error_file = f'prediction_errors{error_part}_zp.json'
                
            print(f"将结果保存到: {output_file}")
            print(f"错误日志将保存到: {error_file}")
            
            # 预分配与输入文件行数相同的结果列表
            results = [None] * input_row_count
            error_logs = []
            
            # 如果输出文件存在，则加载已有结果
            if os.path.exists(output_file):
                try:
                    existing_df = pd.read_excel(output_file)
                    print(f"已加载已有结果文件，包含 {len(existing_df)} 行数据")
                    
                    # 将已有结果填充到对应位置，但仅保留start_index之前的结果
                    for _, row in existing_df.iterrows():
                        if 'row_index' in row and pd.notna(row['row_index']):
                            idx = int(row['row_index'])
                            # 只保存start_index之前的结果
                            if 0 <= idx < start_index and idx < len(results):
                                results[idx] = row.to_dict()
                                print(f"恢复已有结果: row_index={idx}")
                except Exception as e:
                    print(f"加载已有结果失败: {e}")
                    # 不影响继续执行
            
            # 记录处理的条数和批次
            processed_count = 0
            batch_count = 0
            
            # 遍历每一行
            for index in range(input_row_count):
                # 跳过start_index之前的所有记录
                if index < start_index:
                    continue
                
                try:
                    row = input_df.iloc[index]
                    print(f"\n处理第 {index + 1}/{input_row_count} 行:")
                    print(f"challenge_id: {row.get('challenge_id')}")
                    print(f"name: {row[name_col]}")
                    
                    # 检查必要字段是否存在
                    if pd.isna(row[content_col]) or str(row[content_col]).strip() == '':
                        error_msg = f"内容为空: {content_col}"
                        print(f"错误: {error_msg}")
                        error_logs.append({
                            'row_index': index,
                            'challenge_id': row.get('challenge_id'),
                            'name': row[name_col],
                            'error_type': 'empty_content',
                            'error_message': error_msg
                        })
                        
                        # 添加空结果
                        empty_result = {
                            'row_index': index,
                            'challenge_id': row.get('challenge_id'),
                            'name': row[name_col],
                            'knowledge_count': 0,
                            'knowledge_ids': [],
                            'knowledge_names': [],
                            'llm_relevance_scores': [],
                            'vector_similarities': [],
                            'combined_scores': [],
                            'evidences': [],
                            'error': error_msg
                        }
                        results[index] = empty_result
                        processed_count += 1
                        batch_count += 1
                        
                        # 如果这批次已经处理了10条，则保存一次
                        if batch_count >= 10:
                            self._save_excel_results(results, error_logs, output_file, error_file, index, input_row_count, batch_count)
                            batch_count = 0
                        
                        continue
                    
                    # 组合习题文本
                    exercise_text = f"习题名称：{row[name_col]}\n习题内容：{row[content_col]}"
                    
                    # 预测知识点
                    prediction = self.predict_single(exercise_text, row.get('challenge_id'))
                    
                    if not prediction['success']:
                        error_msg = prediction['error'] or "未能预测出知识点"
                        print(f"错误: {error_msg}")
                        error_logs.append({
                            'row_index': index,
                            'challenge_id': row.get('challenge_id'),
                            'name': row[name_col],
                            'error_type': 'no_prediction',
                            'error_message': error_msg
                        })
                    
                    # 构建结果记录 - 修改为支持无限制阈值模式
                    result = {
                        'row_index': index,
                        'challenge_id': row.get('challenge_id'),
                        'name': row[name_col],
                        'knowledge_count': len(prediction['knowledge_points']),  # 添加知识点数量字段
                        'knowledge_ids': [p['topic_id'] for p in prediction['knowledge_points']],
                        'knowledge_names': [p['name'] for p in prediction['knowledge_points']],
                        'llm_relevance_scores': [p['llm_relevance'] for p in prediction['knowledge_points']],
                        'vector_similarities': [p['vector_similarity'] for p in prediction['knowledge_points']],
                        'combined_scores': [p['combined_score'] for p in prediction['knowledge_points']],
                        'evidences': [p['evidence'] for p in prediction['knowledge_points']],
                        'error': prediction['error']
                    }
                    
                    # 将结果保存到对应位置
                    results[index] = result
                    processed_count += 1
                    batch_count += 1
                    
                    # 每处理10行保存一次结果和错误日志
                    if batch_count >= 10:
                        self._save_excel_results(results, error_logs, output_file, error_file, index, input_row_count, batch_count)
                        batch_count = 0
                    
                    # 添加随机延时（0.5-1.5秒）
                    delay = random.uniform(0.5, 1.5)
                    print(f"延时 {delay:.2f} 秒...")
                    time.sleep(delay)
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"处理第 {index + 1} 行时出错: {error_msg}")
                    
                    # 记录错误
                    error_logs.append({
                        'row_index': index,
                        'challenge_id': row.get('challenge_id'),
                        'name': row[name_col],
                        'error_type': 'processing_error',
                        'error_message': error_msg
                    })
                    
                    # 记录空结果
                    empty_result = {
                        'row_index': index,
                        'challenge_id': row.get('challenge_id'),
                        'name': row[name_col],
                        'knowledge_count': 0,
                        'knowledge_ids': [],
                        'knowledge_names': [],
                        'llm_relevance_scores': [],
                        'vector_similarities': [],
                        'combined_scores': [],
                        'evidences': [],
                        'error': error_msg
                    }
                    results[index] = empty_result
                    processed_count += 1
                    batch_count += 1
                    
                    # 如果这批次已经处理了10条，则保存一次
                    if batch_count >= 10:
                        self._save_excel_results(results, error_logs, output_file, error_file, index, input_row_count, batch_count)
                        batch_count = 0
            
            # 最终保存结果
            # 过滤None值，只保存有效结果
            valid_results = [r for r in results if r is not None]
            print(f"\n最终有效结果数: {len(valid_results)}/{input_row_count}")
            
            result_df = pd.DataFrame(valid_results)
            result_df.to_excel(output_file, index=False)
            print(f"\n预测完成，结果已保存到: {output_file}")
            print(f"输入文件行数: {input_row_count}, 结果文件行数: {len(result_df)}")
            
            # 保存错误日志
            if error_logs:
                error_df = pd.DataFrame(error_logs)
                error_df.to_excel(error_file, index=False)
                print(f"错误日志已保存到: {error_file}")
                print(f"共发现 {len(error_logs)} 条错误记录")
            
            return result_df
                
        except Exception as e:
            print(f"处理文件时出错: {str(e)}")
            # 确保在主程序出错时也保存错误日志
            if error_logs:
                error_df = pd.DataFrame(error_logs)
                error_df.to_excel(error_file, index=False)
                print(f"错误日志已保存到: {error_file}")
            return pd.DataFrame()
            
    def _save_excel_results(self, results, error_logs, output_file, error_file, current_index, total_count, batch_count):
        """保存Excel结果和错误日志的辅助方法"""
        # 过滤None值
        valid_results = [r for r in results if r is not None]
        print(f"\n当前有效结果数: {len(valid_results)}/{len(results)}")
        
        # 保存结果
        result_df = pd.DataFrame(valid_results)
        result_df.to_excel(output_file, index=False)
        print(f"已保存当前进度到: {output_file}")
        print(f"当前进度：处理了 {current_index + 1}/{total_count} 行，本批次处理了 {batch_count} 条")
        
        # 保存错误日志
        if error_logs:
            error_df = pd.DataFrame(error_logs)
            error_df.to_excel(error_file, index=False)
            print(f"已更新错误日志: {error_file}")
            print(f"当前共有 {len(error_logs)} 条错误记录")

    def _analyze_knowledge_with_candidates(self, exercise_text: str, candidate_ids: List[str], candidate_desc: str) -> Dict[str, Dict]:
        """
        分析习题与候选知识点的相关度
        Args:
            exercise_text: 习题文本
            candidate_ids: 候选知识点ID列表
            candidate_desc: 候选知识点描述文本
        Returns:
            Dict[str, Dict]: 知识点分析结果，格式为 {知识点ID: {相关信息}}
        """
        try:
            messages = [
                {
                    "role": "system", 
                    "content": """你是一个多学科教育专家。请分析这道习题与给定候选知识点的相关度。

对于每个相关的知识点，请按照以下格式输出（不要输出其他任何内容）：

知识点 <知识点ID> (<知识点名称>):
相关度: <0到1之间的数值>
证据: <从习题中提取的关键文本>
解释: <为什么这个知识点相关的简要说明>

示例输出（假设候选列表中有 "1. DNA的复制 (ID: 3163)"）：
知识点 3163 (DNA的复制):
相关度: 0.94
证据: 习题中描述了DNA复制的过程和机制
解释: 习题内容明确涉及DNA复制的基本原理和过程

**重要提示**：
1. <知识点ID> 必须使用候选列表中括号内的ID数字，不要使用序号！
2. 例如：候选列表中 "5. 蛋白质合成 (ID: 1254)"，你应该写 "知识点 1254"，而不是 "知识点 5"
3. 相关度评分要客观，基于习题内容与知识点的实际关联程度
4. 证据要具体，引用习题中的相关文本
5. 解释要简明扼要，说明知识点与习题内容的关联
6. 只输出确实相关的知识点，不要过度关联"""
                },
                {
                    "role": "user",
                    "content": f"请分析以下习题与候选知识点的相关度：\n\n习题内容：\n{exercise_text}\n\n候选知识点列表：\n{candidate_desc}"
                }
            ]
            
            # 根据模型名称选择客户端
            if self.llm_model.startswith('glm'):
                # 使用智谱 AI 客户端
                response = self.zhipu_client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2000
                )
            else:
                # 使用统一的 client（已根据模型选择）
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2000
                )
            
            # 解析响应
            content = response.choices[0].message.content
            
            # 解析知识点分析结果
            result = {}
            current_topic = None
            
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # 匹配知识点ID
                if line.startswith("知识点 "):
                    try:
                        # 提取知识点ID
                        topic_id = line.split("知识点 ")[1].split(" ")[0].split("(")[0].strip()
                        if topic_id not in candidate_ids:
                            print(f"警告: 模型返回了不在候选列表中的知识点ID: {topic_id}")
                            continue
                        
                        current_topic = topic_id
                        result[current_topic] = {}
                    except Exception as e:
                        print(f"解析知识点ID时出错: {line}, 错误: {str(e)}")
                        current_topic = None
                
                # 匹配相关度
                elif current_topic and "相关度:" in line:
                    try:
                        relevance = float(line.split("相关度:")[1].strip())
                        result[current_topic]['relevance'] = relevance
                    except Exception as e:
                        print(f"解析相关度时出错: {line}, 错误: {str(e)}")
                        result[current_topic]['relevance'] = 0.0
                
                # 匹配证据
                elif current_topic and "证据:" in line:
                    evidence = line.split("证据:")[1].strip()
                    result[current_topic]['evidence'] = evidence
                
                # 匹配解释
                elif current_topic and "解释:" in line:
                    explanation = line.split("解释:")[1].strip()
                    result[current_topic]['explanation'] = explanation
            
            print(f"LLM分析完成，找到 {len(result)} 个相关知识点")
            return result
            
        except Exception as e:
            print(f"分析知识点时出错: {str(e)}")
            return {}

    def _save_results(self, results: List[Dict], errors: List[Dict], output_file: str, error_file: str) -> None:
        """
        保存预测结果和错误日志到Excel文件
        Args:
            results: 预测结果列表
            errors: 错误日志列表
            output_file: 结果输出文件路径
            error_file: 错误日志输出文件路径
        """
        try:
            # 保存结果
            if results:
                result_df = pd.DataFrame(results)
                result_df.to_excel(output_file, index=False)
                print(f"已保存 {len(results)} 条结果记录到: {output_file}")
            
            # 保存错误日志
            if errors:
                error_df = pd.DataFrame(errors)
                error_df.to_excel(error_file, index=False)
                print(f"已保存 {len(errors)} 条错误记录到: {error_file}")
                
        except Exception as e:
            print(f"保存结果时出错: {str(e)}")


def truncate_results_file(file_path: str, keep_until_index: int) -> None:
    """
    处理结果文件，只保留指定索引之前的记录
    
    Args:
        file_path: 结果文件路径（Excel文件）
        keep_until_index: 要保留的最大索引值（不包含该索引）
    """
    try:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return
        
        # 读取Excel文件
        df = pd.read_excel(file_path)
        print(f"已读取 {len(df)} 条记录")
        
        # 找出所有行索引小于指定值的行
        if 'row_index' in df.columns:
            # Excel文件来自predict_from_excel的结果
            filtered_df = df[df['row_index'] < keep_until_index]
            print(f"保留 row_index < {keep_until_index} 的记录: {len(filtered_df)} 条")
        else:
            # 尝试使用problem_id和映射文件
            if os.path.exists('problem_id_mapping.json'):
                with open('problem_id_mapping.json', 'r', encoding='utf-8') as f:
                    id_mapping = json.load(f)
                
                # 创建索引列
                df['_index'] = -1
                for idx, row in df.iterrows():
                    if 'problem_id' in row and row['problem_id'] is not None:
                        # 查找这个problem_id对应的序号
                        for map_idx, mapped_id in id_mapping.items():
                            if mapped_id == row['problem_id']:
                                df.at[idx, '_index'] = int(map_idx)
                                break
                
                filtered_df = df[df['_index'] < keep_until_index]
                filtered_df = filtered_df.drop(columns=['_index'])
                print(f"保留索引 < {keep_until_index} 的记录: {len(filtered_df)} 条")
            else:
                # 如果没有映射文件和索引列，则保留前N条记录
                filtered_df = df.iloc[:keep_until_index]
                print(f"保留前 {keep_until_index} 条记录: {len(filtered_df)} 条")
        
        # 保存结果
        backup_file = f"{file_path}.backup"
        df.to_excel(backup_file, index=False)
        print(f"原文件已备份到: {backup_file}")
        
        filtered_df.to_excel(file_path, index=False)
        print(f"已保存处理后的文件: {file_path}，共 {len(filtered_df)} 条记录")
        
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")

def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description='处理过滤结果并预测知识点')
    parser.add_argument('--start', type=int, help='起始行号（从0开始）', default=0)
    parser.add_argument('--threshold', type=float, help='知识点选择的综合评分阈值', default=0.6)
    parser.add_argument('--unlimited', action='store_true', help='使用无限制阈值模式，返回所有超过阈值的知识点')
    parser.add_argument('--truncate', type=int, help='只保留指定索引之前的记录并退出', default=None)
    parser.add_argument('--file', type=str, help='要处理的文件路径', default="problem_formatted.json")
    args = parser.parse_args()
    
    # 处理文件截断
    if args.truncate is not None:
        print(f"\n开始处理文件: {args.file}")
        print(f"将只保留索引小于 {args.truncate} 的记录")
        truncate_results_file(args.file, args.truncate)
        return
    
    # 显示参数配置
    print(f"\n开始处理，参数配置:")
    print(f"起始行号: {args.start}")
    print(f"评分阈值: {args.threshold}")
    print(f"无限制阈值模式: {'已启用' if args.unlimited else '未启用'}")
    if args.unlimited:
        print(f"  将返回所有超过阈值 {args.threshold} 的知识点，不限制数量")
    else:
        print(f"  将返回最多3个超过阈值的知识点（按分数排序）")
    
    # 生成基于模式、阈值和起始行号的输出文件名
    mode = "unlimited" if args.unlimited else "limited"
    threshold_str = str(args.threshold).replace('.', '_')
    
    # 在文件名中添加起始行号标记
    start_part = f"_from{args.start}" if args.start > 0 else ""
    output_file = f"prediction_results_{mode}_threshold{threshold_str}{start_part}_zp.xlsx"
    error_file = f"prediction_errors{start_part}_zp.json"
    
    print(f"结果将保存至: {output_file}")
    print(f"错误日志将保存至: {error_file}")
    print()
    
    predictor = GPTKnowledgePredictor(
        analysis_top_k=30,  # 分析阶段考虑30个候选知识点
        result_top_k=3,    # 最终返回3个最相关的知识点
        score_threshold=args.threshold,  # 使用命令行参数设置阈值
        unlimited_threshold=args.unlimited,  # 使用命令行参数设置无限制阈值模式
    )
    
    # 从JSON文件预测知识点
    predictor.predict_from_json(
        json_file=args.file,
        output_file=output_file,
        error_file=error_file,
        start_index=args.start
    )

if __name__ == "__main__":
    main()