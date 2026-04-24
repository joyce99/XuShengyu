import os
import json
import config
import pandas as pd
from typing import List, Dict, Union, Tuple
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import faiss
import pickle
from pathlib import Path
import time
import random

class GPTKnowledgePredictor:
    def __init__(self, topics_file: str = "topics.csv", 
                 index_dir: str = "vector_index",
                 analysis_top_k: int = 10,    # 分析阶段的候选知识点数量
                 result_top_k: int = 3,      # 最终结果的知识点数量
                 score_threshold: float = 0.5,  # 知识点选择的综合评分阈值
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
            
        # 初始化OpenAI客户端 - GPT模型使用
        self.client = OpenAI(
            api_key=gpt_api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=gpt_base_url or os.environ.get("OPENAI_BASE_URL"),
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

    def _analyze_knowledge_aspects(self, exercise_text: str) -> Dict[str, Dict]:
        """
        分析练习文本涉及的知识点
        Returns:
            Dict[str, Dict]: 知识点分析结果
        """
        # 首先用embedding找出最相关的知识点
        query_vector = self._get_embedding(f"编程习题：{exercise_text}")
        faiss.normalize_L2(query_vector.reshape(1, -1))
        
        # 搜索最相似的前10个知识点
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
                "content": """你是一个编程教育专家。请分析这道编程练习涉及的知识点。

对于每个相关的知识点，请按照以下格式输出（不要输出其他任何内容）：

知识点 <知识点ID> (<知识点名称>):
相关度: <0到1之间的数值>
证据: <从习题中提取的关键文本>
解释: <为什么这个知识点相关的简要说明>

示例输出：
知识点 3163 (创建数据库实例):
相关度: 0.94
证据: 数据库连接与数据库实例创建
解释: 习题中明确提到了创建数据库实例的操作

注意：
1. 只分析真正相关的知识点
2. 相关度必须是0-1之间的数值
3. 证据应该来自习题原文
4. 解释要简洁明了"""
            },
            {
                "role": "user", 
                "content": f"""请分析这道编程练习涉及哪些知识点。

习题内容：
{exercise_text}

候选知识点：
{json.dumps(candidate_points, indent=2, ensure_ascii=False)}"""
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model="qwen2.5-coder-14b-instruct",
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # 解析输出格式
            result = {}
            current_topic = None
            current_data = {}
            
            for line in response_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('知识点 '):
                    # 如果有前一个知识点的数据，保存它
                    if current_topic and current_data:
                        result[current_topic] = current_data
                        current_data = {}
                    
                    # 解析新的知识点
                    parts = line.split(' ', 2)
                    topic_id = parts[1]
                    name = parts[2].strip('():')
                    current_topic = topic_id
                    current_data['name'] = name
                    
                elif line.startswith('相关度:'):
                    current_data['relevance'] = float(line.split(':', 1)[1].strip())
                elif line.startswith('证据:'):
                    current_data['evidence'] = line.split(':', 1)[1].strip()
                elif line.startswith('解释:'):
                    current_data['explanation'] = line.split(':', 1)[1].strip()
            
            # 保存最后一个知识点的数据
            if current_topic and current_data:
                result[current_topic] = current_data
            
            # 验证结果
            if not result:
                print("解析结果为空")
                return self._get_fallback_result(similarities, indices)
            
            print("\n知识点分析结果：")
            for topic_id, info in result.items():
                print(f"\n知识点 {topic_id} ({info['name']}):")
                print(f"相关度: {info['relevance']}")
                print(f"证据: {info['evidence']}")
                print(f"解释: {info['explanation']}")
            
            return result
            
        except Exception as e:
            print(f"知识点分析失败: {str(e)}")
            return self._get_fallback_result(similarities, indices)

    def _get_knowledge_based_embeddings(self, exercise_text: str) -> Dict[int, List[Dict]]:
        """
        基于知识点分析获取向量表示
        Returns:
            Dict[int, List[Dict]]: 知识点ID对应的相关文本向量信息
        """
        analysis = self._analyze_knowledge_aspects(exercise_text)
        
        embeddings = {}
        for topic_id, info in analysis.items():
            vector = self._get_embedding(info['evidence'])
            if len(vector) > 0:
                embeddings[int(topic_id)] = [{
                    'vector': vector,
                    'text': info['evidence'],
                    'relevance': float(info['relevance'])
                }]
                
        return embeddings

    def _search_with_knowledge_embeddings(
        self, 
        embeddings: Dict[int, List[Dict]]
    ) -> List[Tuple[int, float, str]]:
        """
        使用基于知识点的向量进行相似度搜索
        Args:
            embeddings: 知识点的向量表示
        Returns:
            List[Tuple[int, float, str]]: [(知识点ID, 综合得分, 最相关文本)]
        """
        results = []
        
        for topic_id, vectors_info in embeddings.items():
            for vec_info in vectors_info:
                query_vector = vec_info['vector']
                faiss.normalize_L2(query_vector.reshape(1, -1))
                
                similarities, _ = self.index.search(query_vector.reshape(1, -1), 1)
                similarity = float(similarities[0][0])
                
                combined_score = similarity * vec_info['relevance']
                
                # 只添加超过阈值的结果
                if combined_score >= self.score_threshold:
                    results.append((topic_id, combined_score, vec_info['text']))
        
        # 根据 score_threshold 是否为默认值决定返回策略
        if hasattr(self, '_custom_threshold_set') and self._custom_threshold_set:
            # 如果设置了自定义阈值，则返回所有超过阈值的结果（按分数排序）
            return sorted(results, key=lambda x: x[1], reverse=True)
        else:
            # 否则按照原来的逻辑，返回前 result_top_k 个结果
            return sorted(results, key=lambda x: x[1], reverse=True)[:self.result_top_k]

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
            
            for topic_id, combined_score, evidence in similar_topics:
                # 从knowledge_embeddings中获取相关度
                topic_info = next(iter(knowledge_embeddings[topic_id]))
                relevance = topic_info['relevance']
                # 计算向量相似度（综合得分 / 相关度）
                vector_similarity = combined_score / relevance if relevance > 0 else 0
                
                point_info = {
                    'topic_id': topic_id,
                    'name': self.knowledge_points[topic_id],
                    'relevance': relevance,  # LLM分析的相关度
                    'vector_similarity': vector_similarity,  # 向量相似度
                    'combined_score': combined_score,  # 综合得分
                    'evidence': evidence  # 关键证据
                }
                result_points.append(point_info)
                
                print(f"\n知识点 {topic_id}: {self.knowledge_points[topic_id]}")
                print(f"LLM相关度: {relevance:.3f}")
                print(f"向量相似度: {vector_similarity:.3f}")
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
                          start_index: int = 0) -> pd.DataFrame:
        """从 Excel 文件批量预测知识点"""
        try:
            # 读取输入文件
            input_df = pd.read_excel(excel_file)
            input_row_count = len(input_df)
            print(f"成功读取文件: {excel_file}, 总行数: {input_row_count}")
            
            # 存储结果和错误的列表
            results = []
            error_logs = []
            
            # 添加模式信息到输出文件名
            mode_suffix = "_unlimited" if hasattr(self, '_custom_threshold_set') and self._custom_threshold_set else ""
            output_file = f'knowledge_prediction_results{mode_suffix}.xlsx'
            error_file = f'knowledge_prediction_errors{mode_suffix}.xlsx'
            
            # 如果不是从头开始，且输出文件存在，则加载已有结果
            if start_index > 0 and os.path.exists(output_file):
                try:
                    existing_df = pd.read_excel(output_file)
                    results = existing_df.to_dict('records')
                    print(f"已加载 {len(results)} 条已处理记录")
                    
                    # 验证已有结果的行数是否与输入文件匹配
                    if len(existing_df) != input_row_count:
                        print(f"警告：已有结果行数({len(existing_df)})与输入文件行数({input_row_count})不匹配")
                        # 清空结果重新处理
                        results = []
                        start_index = 0
                except Exception as e:
                    print(f"加载已有结果失败: {e}")
            
            # 遍历每一行
            for index in range(start_index, input_row_count):
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
                            'relevance_scores': [],
                            'vector_similarities': [],
                            'combined_scores': [],
                            'evidences': [],
                            'error': error_msg
                        }
                        if index < len(results):
                            results[index] = empty_result
                        else:
                            results.append(empty_result)
                        print(f"添加空结果后列表长度: {len(results)}")
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
                        'relevance_scores': [p['relevance'] for p in prediction['knowledge_points']],
                        'vector_similarities': [p['vector_similarity'] for p in prediction['knowledge_points']],
                        'combined_scores': [p['combined_score'] for p in prediction['knowledge_points']],
                        'evidences': [p['evidence'] for p in prediction['knowledge_points']],
                        'error': prediction['error']
                    }
                    
                    # 如果是从中间开始的，确保结果插入到正确的位置
                    if index < len(results):
                        results[index] = result
                        print(f"更新第 {index} 行结果")
                    else:
                        results.append(result)
                        print(f"添加新结果到第 {index} 行")
                    print(f"添加/更新结果后列表长度: {len(results)}")
                    
                    # 每处理10行保存一次结果和错误日志
                    if (index + 1) % 10 == 0:
                        # 保存结果
                        result_df = pd.DataFrame(results)
                        result_df.to_excel(output_file, index=False)
                        print(f"已保存当前进度到: {output_file}")
                        print(f"当前进度：处理了 {index + 1} 行，结果文件有 {len(result_df)} 行")
                        
                        # 保存错误日志
                        if error_logs:
                            error_df = pd.DataFrame(error_logs)
                            error_df.to_excel(error_file, index=False)
                            print(f"已更新错误日志: {error_file}")
                    
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
                    
                    # 记录空结果（保持行数一致）
                    empty_result = {
                        'row_index': index,
                        'challenge_id': row.get('challenge_id'),
                        'name': row[name_col],
                        'knowledge_count': 0,
                        'knowledge_ids': [],
                        'knowledge_names': [],
                        'relevance_scores': [],
                        'vector_similarities': [],
                        'combined_scores': [],
                        'evidences': [],
                        'error': error_msg
                    }
                    
                    if index < len(results):
                        results[index] = empty_result
                        print(f"更新错误结果到第 {index} 行")
                    else:
                        results.append(empty_result)
                        print(f"添加错误结果到第 {index} 行")
                    print(f"添加错误结果后列表长度: {len(results)}")
            
            # 最终保存结果前验证行数
            result_df = pd.DataFrame(results)
            print(f"\n最终行数检查:")
            print(f"输入文件行数: {input_row_count}")
            print(f"结果列表长度: {len(results)}")
            print(f"结果DataFrame行数: {len(result_df)}")
            
            if len(result_df) != input_row_count:
                print(f"警告：最终结果行数({len(result_df)})与输入文件行数({input_row_count})不匹配")
                print("开始补充缺失的行...")
                # 确保行数一致
                if len(result_df) < input_row_count:
                    # 补充缺失的行
                    for i in range(len(result_df), input_row_count):
                        print(f"补充第 {i} 行")
                        results.append({
                            'row_index': i,
                            'challenge_id': input_df.iloc[i].get('challenge_id'),
                            'name': input_df.iloc[i][name_col],
                            'knowledge_count': 0,
                            'knowledge_ids': [],
                            'knowledge_names': [],
                            'relevance_scores': [],
                            'vector_similarities': [],
                            'combined_scores': [],
                            'evidences': [],
                            'error': '数据缺失'
                        })
                    result_df = pd.DataFrame(results)
                    print(f"补充后的行数: {len(result_df)}")
            
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

def main():
    """测试运行"""
    import argparse
    parser = argparse.ArgumentParser(description='处理过滤结果并预测知识点')
    parser.add_argument('--start', type=int, help='起始行号（从0开始）', default=0)
    parser.add_argument('--threshold', type=float, help='知识点选择的综合评分阈值', default=0.6)
    parser.add_argument('--unlimited', action='store_true', help='使用无限制阈值模式，返回所有超过阈值的知识点')
    args = parser.parse_args()
    
    # 显示参数配置
    print(f"\n开始处理，参数配置:")
    print(f"起始行号: {args.start}")
    print(f"评分阈值: {args.threshold}")
    print(f"无限制阈值模式: {'已启用' if args.unlimited else '未启用'}")
    if args.unlimited:
        print(f"  将返回所有超过阈值 {args.threshold} 的知识点，不限制数量")
    else:
        print(f"  将返回最多3个超过阈值的知识点（按分数排序）")
    print()
    
    predictor = GPTKnowledgePredictor(
        analysis_top_k=10,  # 分析阶段考虑5个候选知识点
        result_top_k=3,    # 最终返回3个最相关的知识点
        score_threshold=args.threshold,  # 使用命令行参数设置阈值
        unlimited_threshold=args.unlimited,  # 使用命令行参数设置无限制阈值模式
    )
    
    # 从Excel文件预测知识点
    results = predictor.predict_from_excel(
        excel_file="filtered_results_with_summary.xlsx",
        name_col="name",
        content_col="summarized_content",
        start_index=args.start
    )

if __name__ == "__main__":
    main()