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
    def __init__(self, topics_file: str = "data/xlsx/topics.csv", 
                 index_dir: str = "data/vector_index",
                 analysis_top_k: int = 30,    # 分析阶段的候选知识点数量
                 result_top_k: int = 3,      # 最终结果的知识点数量
                 score_threshold: float = 0.6,  # 知识点选择的综合评分阈值
                 gpt_api_key: str = None,    # LLM API密钥
                 gpt_base_url: str = None,   # LLM API地址
                 embedding_api_key: str = None,  # 嵌入模型API密钥
                 embedding_base_url: str = None,  # 嵌入模型API地址
                 unlimited_threshold: bool = False  # 是否使用无限制阈值模式
                 ):
        """
        初始化预测器
        Args:
            topics_file: 知识点CSV文件路径
            index_dir: 向量索引保存目录
            analysis_top_k: 分析阶段的候选知识点数量
            result_top_k: 最终返回的知识点数量
            score_threshold: 知识点选择的综合评分阈值，只有超过此阈值的知识点才会被选择
            gpt_api_key: GPT API密钥，如果为None则使用环境变量
            gpt_base_url: GPT API地址，如果为None则使用环境变量
            embedding_api_key: 嵌入模型API密钥，如果为None则使用环境变量
            embedding_base_url: 嵌入模型API地址，如果为None则使用环境变量
            unlimited_threshold: 设置为True时，将返回所有超过阈值的知识点，而不限制数量
        """
        if not os.path.exists(topics_file):
            raise FileNotFoundError(f"找不到知识点文件: {topics_file}")
            
        self.analysis_top_k = analysis_top_k
        self.result_top_k = result_top_k
        self.score_threshold = score_threshold
        
        # 标记是否使用自定义阈值模式（不限制返回数量）
        self._custom_threshold_set = unlimited_threshold
            
        # 初始化ZhipuAI客户端 - GLM模型使用
        self.client = ZhipuAI(
            api_key=gpt_api_key or os.environ.get("ZHIPUAI_API_KEY"),
        )
        
        # 初始化OpenAI客户端 - 嵌入模型使用
        self.embedding_client = OpenAI(
            api_key=embedding_api_key or os.environ.get("EMBEDDING_API_KEY"),
            base_url=embedding_base_url or os.environ.get("EMBEDDING_BASE_URL"),
        )
            
        # 读取CSV文件
        self.topics_df = pd.read_csv(topics_file)
        self.knowledge_points = dict(zip(
            self.topics_df.topic_id, 
            self.topics_df.topic_name
        ))
        
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
            topic_desc = f"编程知识点：{topic_name}"
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
                topic_desc = f"编程知识点：{topic_name}"
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
            
            # 使用LLM分析习题与候选知识点的相关度（将candidate_ids转为字符串列表）
            candidate_ids_str = [str(c_id) for c_id in candidate_ids]
            llm_results = self._analyze_knowledge_with_candidates(exercise_text, candidate_ids_str, candidate_desc)

            # 4. 整合向量相似度和LLM分析结果
            knowledge_embeddings = {}
            for topic_id in candidate_ids:
                # 获取向量相似度
                vector_similarity = vector_similarities[topic_id]
                
                # 获取LLM分析的相关度（如果有）- LLM返回的是字符串ID
                topic_id_str = str(topic_id)
                llm_relevance = 0.0
                llm_evidence = ""
                if topic_id_str in llm_results:
                    llm_relevance = llm_results[topic_id_str].get('relevance', 0.0)
                    llm_evidence = llm_results[topic_id_str].get('evidence', "")
                
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
            llm_weight = 0.0  # LLM相关度权重
            vector_weight = 1.0  # 向量相似度权重
            
            for topic_id, vectors_info in embeddings.items():
                for vec_info in vectors_info:
                    # 获取两个分数
                    llm_relevance = vec_info['llm_relevance']  # LLM相关度
                    vector_similarity = vec_info['vector_similarity']  # 向量相似度
                    evidence = vec_info['evidence']
                    
                    # 计算加权综合得分
                    combined_score = (llm_relevance * llm_weight) + (vector_similarity * vector_weight)
                    
                    # 调试信息：显示每个候选知识点的得分
                    print(f"  候选 {topic_id}: LLM={llm_relevance:.3f}, Vec={vector_similarity:.3f}, 综合={combined_score:.3f}, 阈值={self.score_threshold}")
                    
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
                    "content": """你是一个编程教育专家。请分析这道习题与给定候选知识点的相关度。

对于每个相关的知识点，请按照以下格式输出（不要输出其他任何内容）：

知识点 <知识点ID> (<知识点名称>):
相关度: <0到1之间的数值>
证据: <从习题中提取的关键文本>
解释: <为什么这个知识点相关的简要说明>

示例输出（假设候选列表中有 "1. 创建数据库实例 (ID: 3163)"）：
知识点 3163 (创建数据库实例):
相关度: 0.94
证据: 数据库连接与数据库实例创建
解释: 习题中明确提到了创建数据库实例的操作

**重要提示**：
1. <知识点ID> 必须使用候选列表中括号内的ID数字，不要使用序号！
2. 例如：候选列表中 "5. 循环语句 (ID: 1254)"，你应该写 "知识点 1254"，而不是 "知识点 5"
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
            
            # 调用GPT获取分析结果
            response = self.client.chat.completions.create(
                model="glm-4-air-250414",
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

    def predict_single(self, exercise_text: str, challenge_id: str = None) -> Dict:
        """
        预测单个习题的知识点
        Returns:
            Dict: {
                'success': bool,  # 是否成功预测
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
            llm_weight = 0.0  # LLM相关度权重
            vector_weight = 1.0  # 向量相似度权重
            
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

    def predict_from_excel(self, excel_file: str, name_col: str = 'name', 
                          content_col: str = 'summarized_content', 
                          start_index: int = 0,
                          output_file: str = None,
                          error_file: str = None) -> pd.DataFrame:
        """
        从 Excel 文件中读取习题，预测知识点并返回结果
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
                error_file = f'prediction_errors{error_part}_zp.xlsx'
                
            print(f"将结果保存到: {output_file}")
            print(f"错误日志将保存到: {error_file}")
            
            # 计算需要处理的行数
            rows_to_process = input_row_count - start_index
            
            # 预分配结果列表（从0开始计数）
            results = []
            error_logs = []
            
            # 记录处理的条数和批次
            processed_count = 0
            batch_count = 0
            output_row_index = 0  # 输出文件的行号从0开始
            
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
                    print(f"当前结果列表长度: {len(results)}")
                    
                    # 检查必要字段是否存在
                    if pd.isna(row[content_col]) or str(row[content_col]).strip() == '':
                        error_msg = f"内容为空: {content_col}"
                        print(f"错误: {error_msg}")
                        error_logs.append({
                            'row_index': output_row_index,
                            'original_index': index,
                            'challenge_id': row.get('challenge_id'),
                            'name': row[name_col],
                            'error_type': 'empty_content',
                            'error_message': error_msg
                        })
                        # 添加空结果
                        empty_result = {
                            'row_index': output_row_index,
                            'original_index': index,
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
                        results.append(empty_result)
                        output_row_index += 1
                        processed_count += 1
                        batch_count += 1
                        
                        # 每处理1条保存一次
                        self._save_excel_results(results, error_logs, output_file, error_file, output_row_index - 1, rows_to_process, batch_count)
                        
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
                    
                    # 构建结果记录 - 支持无限制阈值模式
                    result = {
                        'row_index': output_row_index,  # 输出文件中的行号（从0开始）
                        'original_index': index,  # 原始数据中的行号
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
                    
                    # 将结果添加到列表
                    results.append(result)
                    output_row_index += 1
                    processed_count += 1
                    batch_count += 1
                    
                    # 每处理1行保存一次结果（改为每条都保存，防止数据丢失）
                    self._save_excel_results(results, error_logs, output_file, error_file, output_row_index - 1, rows_to_process, batch_count)
                    
                    # 添加随机延时（0.5-1.5秒）
                    delay = random.uniform(0.5, 1.5)
                    print(f"延时 {delay:.2f} 秒...")
                    time.sleep(delay)
                    
                except KeyboardInterrupt:
                    print("\n用户中断处理！正在保存当前结果...")
                    # 保存当前进度
                    result_df = pd.DataFrame(results)
                    result_df.to_excel(output_file, index=False)
                    if error_logs:
                        error_df = pd.DataFrame(error_logs)
                        error_df.to_excel(error_file, index=False)
                    print(f"已保存中断时的进度，完成到第 {index} 行")
                    print(f"下次可以使用 --start {index} 从此处继续")
                    return result_df
                
                except Exception as e:
                    error_msg = str(e)
                    print(f"处理第 {index + 1} 行时出错: {error_msg}")
                    
                    # 记录错误
                    error_logs.append({
                        'row_index': output_row_index,
                        'original_index': index,
                        'challenge_id': row.get('challenge_id'),
                        'name': row[name_col],
                        'error_type': 'processing_error',
                        'error_message': error_msg
                    })
                    
                    # 记录空结果
                    empty_result = {
                        'row_index': output_row_index,
                        'original_index': index,
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
                    results.append(empty_result)
                    output_row_index += 1
                    processed_count += 1
                    batch_count += 1
                    
                    # 每处理1条保存一次
                    self._save_excel_results(results, error_logs, output_file, error_file, output_row_index - 1, rows_to_process, batch_count)
            
            # 最终保存结果
            print(f"\n最终有效结果数: {len(results)}/{rows_to_process}")
            
            # 使用统一的保存方法
            if results:
                self._save_excel_results(results, error_logs, output_file, error_file, len(results) - 1, rows_to_process, len(results))
            
            print(f"\n预测完成，结果已保存到: {output_file}")
            print(f"需要处理的行数: {rows_to_process}, 结果文件行数: {len(results)}")
            
            # 保存错误日志
            if error_logs:
                error_df = pd.DataFrame(error_logs)
                error_df.to_excel(error_file, index=False)
                print(f"错误日志已保存到: {error_file}")
                print(f"共发现 {len(error_logs)} 条错误记录")
            
            return pd.DataFrame(results)
                
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
        
        if not valid_results:
            print("警告：没有有效结果可保存")
            return
        
        # 将列表字段转换为字符串，便于Excel显示
        formatted_results = []
        for r in valid_results:
            formatted_row = {
                'row_index': r.get('row_index'),
                'original_index': r.get('original_index'),  # 保留原始行号便于追溯
                'challenge_id': r.get('challenge_id'),
                'name': r.get('name'),
                'knowledge_count': r.get('knowledge_count', 0),
                'knowledge_ids': ', '.join(map(str, r.get('knowledge_ids', []))) if r.get('knowledge_ids') else '',
                'knowledge_names': ', '.join(r.get('knowledge_names', [])) if r.get('knowledge_names') else '',
                'llm_relevance_scores': ', '.join(f'{x:.4f}' for x in r.get('llm_relevance_scores', [])) if r.get('llm_relevance_scores') else '',
                'vector_similarities': ', '.join(f'{x:.4f}' for x in r.get('vector_similarities', [])) if r.get('vector_similarities') else '',
                'combined_scores': ', '.join(f'{x:.4f}' for x in r.get('combined_scores', [])) if r.get('combined_scores') else '',
                'evidences': ' | '.join(r.get('evidences', [])) if r.get('evidences') else '',
                'error': r.get('error')
            }
            formatted_results.append(formatted_row)
        
        # 保存结果
        result_df = pd.DataFrame(formatted_results)
        result_df.to_excel(output_file, index=False)
        print(f"已保存当前进度到: {output_file}")
        print(f"当前进度：处理了 {current_index + 1}/{total_count} 行，本批次处理了 {batch_count} 条")
        
        # 保存错误日志
        if error_logs:
            error_df = pd.DataFrame(error_logs)
            error_df.to_excel(error_file, index=False)
            print(f"已更新错误日志: {error_file}")
            print(f"当前共有 {len(error_logs)} 条错误记录")

def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description='处理过滤结果并预测知识点')
    parser.add_argument('--start', type=int, help='起始行号（从0开始）', default=0)
    parser.add_argument('--threshold', type=float, help='知识点选择的综合评分阈值', default=0.6)
    parser.add_argument('--unlimited', action='store_true', help='使用无限制阈值模式，返回所有超过阈值的知识点')
    parser.add_argument('--input', type=str, help='输入Excel文件路径', default="data/xlsx/filtered_results_with_summary.xlsx")
    parser.add_argument('--name_col', type=str, help='习题名称列名', default="name")
    parser.add_argument('--content_col', type=str, help='习题内容列名', default="summarized_content")
    parser.add_argument('--top_k', type=int, help='返回的知识点数量', default=3)
    parser.add_argument('--analysis_k', type=int, help='分析阶段的候选知识点数量', default=30)
    args = parser.parse_args()
    
    # 显示参数配置
    print(f"\n开始处理，参数配置:")
    print(f"起始行号: {args.start}")
    print(f"评分阈值: {args.threshold}")
    print(f"无限制阈值模式: {'已启用' if args.unlimited else '未启用'}")
    if args.unlimited:
        print(f"  将返回所有超过阈值 {args.threshold} 的知识点，不限制数量")
    else:
        print(f"  将返回最多{args.top_k}个超过阈值的知识点（按分数排序）")
    print(f"输入文件: {args.input}")
    print(f"名称列: {args.name_col}")
    print(f"内容列: {args.content_col}")
    print(f"候选知识点数量: {args.analysis_k}\n")
    
    predictor = GPTKnowledgePredictor(
        analysis_top_k=args.analysis_k,  # 分析阶段考虑的候选知识点数量
        result_top_k=args.top_k,    # 最终返回的知识点数量
        score_threshold=args.threshold,  # 使用命令行参数设置阈值
        unlimited_threshold=args.unlimited,  # 使用命令行参数设置无限制阈值模式
    )
    
    # 从Excel文件预测知识点
    results = predictor.predict_from_excel(
        excel_file=args.input,
        name_col=args.name_col,
        content_col=args.content_col,
        start_index=args.start
    )

if __name__ == "__main__":
    main()