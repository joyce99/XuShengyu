# -*- coding: utf-8 -*-
"""
知识图谱构建模块 - Mooper数据集版本

使用 LLM 为知识点构建层级关系和组合关系图谱。

两阶段构建策略：
1. 阶段1：对所有知识点进行领域分类
2. 阶段2：在同一领域内分析组合关系
"""

import os
import json
import time
import pandas as pd
from typing import List, Dict, Tuple
from openai import OpenAI
from tqdm import tqdm
import config


class KnowledgeGraphBuilder:
    """基于LLM的知识图谱构建器"""
    
    def __init__(self, topics_file: str = "data/xlsx/topics.csv",
                 output_path: str = "data/knowledge_graph.json",
                 batch_size: int = 30,
                 api_key: str = None,
                 base_url: str = None):
        """
        初始化构建器
        
        Args:
            topics_file: 知识点CSV文件路径
            output_path: 输出知识图谱文件路径
            batch_size: 每次LLM调用处理的知识点数量
        """
        self.topics_file = topics_file
        self.output_path = output_path
        self.batch_size = batch_size
        
        # 加载知识点 (从CSV)
        topics_df = pd.read_csv(topics_file)
        self.concept_mapping = dict(zip(
            topics_df.topic_id.astype(str),
            topics_df.topic_name
        ))
        
        # 构建反向映射
        self.reverse_mapping = {v: k for k, v in self.concept_mapping.items()}
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key or os.environ.get("DASHSCOPE_API_KEY"),
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 知识图谱结构
        self.knowledge_graph = {
            "hierarchy": {},
            "composites": {},
            "domains": {},
            "metadata": {
                "total_concepts": len(self.concept_mapping),
                "build_time": None
            }
        }
        
        print(f"已加载 {len(self.concept_mapping)} 个知识点")
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """调用LLM并返回结果"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": "你是一个知识图谱专家，擅长分析编程知识点之间的层级关系和组合关系。请严格按照要求的JSON格式输出。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=4096
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"LLM调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return ""
    
    def _parse_llm_json(self, response: str) -> dict:
        """解析LLM返回的JSON"""
        import re
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return {}

    def analyze_composite_relationships_in_domain(self, concept_names: List[str], 
                                                    domain: str) -> Dict:
        """分析同一领域内知识点的组合关系"""
        concepts_str = "\n".join([f"- {name}" for name in concept_names])
        
        prompt = f"""你是知识图谱专家。以下是"{domain}"领域的编程知识点，请分析它们之间的组合关系。

知识点列表：
{concepts_str}

请找出这些知识点中，哪些可以两两组合（或多个组合）形成一个更通用/更高层级的知识点。

组合关系的例子：
- "for循环" + "while循环" → "循环语句"
- "if语句" + "switch语句" → "条件语句"
- "数组" + "链表" → "线性数据结构"
- "继承" + "多态" → "面向对象特性"

请以JSON格式输出：
{{
    "composites": [
        {{
            "components": ["知识点A", "知识点B"],
            "result": "组合后的知识点",
            "confidence": 0.9
        }}
    ]
}}

注意：
1. components 中的知识点必须来自上面的列表
2. result 是组合后的更高层级概念
3. confidence 是置信度 (0-1)，只输出 confidence >= 0.7 的组合
4. 不要强行组合无关的知识点
5. 只输出JSON"""

        response = self._call_llm(prompt)
        return self._parse_llm_json(response)

    def _classify_domains(self, concepts: List[str]) -> Dict[str, List[str]]:
        """使用LLM对知识点进行领域分类"""
        domains = {}
        batch_size = 50
        
        for i in tqdm(range(0, len(concepts), batch_size), desc="领域分类"):
            batch = concepts[i:i + batch_size]
            concepts_str = "\n".join([f"- {c}" for c in batch])
            
            prompt = f"""将以下编程相关知识点按领域分类：

{concepts_str}

请以JSON格式输出：
{{
    "领域名称1": ["知识点1", "知识点2"],
    "领域名称2": ["知识点3", "知识点4"]
}}

领域示例：基础语法、面向对象、数据结构、算法、数据库、文件操作、网络编程、计算机基础等。
只输出JSON。"""
            
            response = self._call_llm(prompt)
            result = self._parse_llm_json(response)
            
            for domain, domain_concepts in result.items():
                if domain not in domains:
                    domains[domain] = []
                domains[domain].extend(domain_concepts)
            
            time.sleep(0.3)
        
        return domains

    def _store_composite_relationship(self, comp: Dict):
        """存储单个组合关系"""
        components = comp.get("components", [])
        result_name = comp.get("result", "")
        
        if len(components) < 2 or not result_name:
            return
        
        result_id = self.reverse_mapping.get(result_name)
        
        component_ids = []
        for c in components:
            cid = self.reverse_mapping.get(c)
            if cid:
                component_ids.append(cid)
        
        if len(component_ids) < 2:
            return
        
        if not result_id:
            result_id = f"_composite_{result_name}"
        
        if result_id not in self.knowledge_graph["composites"]:
            self.knowledge_graph["composites"][result_id] = {
                "name": result_name,
                "component_sets": []
            }
        
        sorted_components = sorted(component_ids)
        if sorted_components not in self.knowledge_graph["composites"][result_id]["component_sets"]:
            self.knowledge_graph["composites"][result_id]["component_sets"].append(sorted_components)

    def _analyze_domain_composites(self, concepts: List[str], domain: str):
        """分析单个领域内的组合关系"""
        result = self.analyze_composite_relationships_in_domain(concepts, domain)
        
        if result and "composites" in result:
            for comp in result["composites"]:
                self._store_composite_relationship(comp)

    def _get_sub_groups(self, concepts: List[str], domain: str) -> Dict[str, List[str]]:
        """让LLM对领域内的知识点进行子主题分组"""
        concepts_str = "\n".join([f"- {c}" for c in concepts])
        
        prompt = f"""将以下"{domain}"领域的知识点按子主题分组：

{concepts_str}

请将这些知识点按照更细分的子主题分组，每组应包含语义相关的知识点。

以JSON格式输出：
{{
    "子主题1": ["知识点1", "知识点2"],
    "子主题2": ["知识点3", "知识点4"]
}}

只输出JSON。"""
        
        response = self._call_llm(prompt)
        result = self._parse_llm_json(response)
        
        if not result:
            return {domain: concepts}
        
        return result

    def _analyze_large_domain_composites(self, concepts: List[str], domain: str):
        """对大领域使用智能分批策略"""
        sub_groups = self._get_sub_groups(concepts, domain)
        
        for sub_topic, sub_concepts in sub_groups.items():
            if len(sub_concepts) < 2:
                continue
            
            if len(sub_concepts) <= self.batch_size:
                self._analyze_domain_composites(sub_concepts, f"{domain}/{sub_topic}")
            else:
                overlap = self.batch_size // 3
                for i in range(0, len(sub_concepts), self.batch_size - overlap):
                    batch = sub_concepts[i:i + self.batch_size]
                    if len(batch) >= 2:
                        self._analyze_domain_composites(batch, f"{domain}/{sub_topic}")
                    time.sleep(0.3)

    def build_graph_by_domain(self):
        """按领域分批构建知识图谱（推荐方法）"""
        all_concepts = list(self.concept_mapping.values())
        
        print("="*60)
        print("阶段1: 对知识点进行领域分类...")
        print("="*60)
        
        domains = self._classify_domains(all_concepts)
        
        for domain, concepts in domains.items():
            concept_ids = [self.reverse_mapping.get(c) for c in concepts if c in self.reverse_mapping]
            concept_ids = [cid for cid in concept_ids if cid]
            if concept_ids:
                self.knowledge_graph["domains"][domain] = concept_ids
        
        print(f"\n分类完成！共 {len(domains)} 个领域:")
        for domain, concepts in sorted(domains.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  - {domain}: {len(concepts)} 个知识点")
        
        print("\n" + "="*60)
        print("阶段2: 在各领域内分析组合关系...")
        print("="*60)
        
        for domain, concepts in tqdm(domains.items(), desc="处理领域"):
            if len(concepts) < 2:
                continue
            
            if len(concepts) <= self.batch_size:
                self._analyze_domain_composites(concepts, domain)
            else:
                self._analyze_large_domain_composites(concepts, domain)
            
            time.sleep(0.3)
        
        self._save_graph()
        return self.knowledge_graph

    def _save_graph(self):
        """保存知识图谱到文件"""
        import datetime
        self.knowledge_graph["metadata"]["build_time"] = datetime.datetime.now().isoformat()
        
        os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else '.', exist_ok=True)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_graph, f, ensure_ascii=False, indent=2)
        
        print(f"知识图谱已保存至: {self.output_path}")

    def load_existing_graph(self) -> bool:
        """加载已存在的知识图谱"""
        if os.path.exists(self.output_path):
            with open(self.output_path, 'r', encoding='utf-8') as f:
                self.knowledge_graph = json.load(f)
            print(f"已加载现有知识图谱: {len(self.knowledge_graph.get('composites', {}))} 个组合关系")
            return True
        return False

    def get_composite_rules(self) -> Dict[Tuple[str, str], str]:
        """将知识图谱转换为组合规则表"""
        rules = {}
        
        for result_id, data in self.knowledge_graph.get("composites", {}).items():
            if result_id.startswith("_composite_"):
                continue
            
            for component_set in data.get("component_sets", []):
                if len(component_set) == 2:
                    key = tuple(sorted(component_set))
                    rules[key] = result_id
                elif len(component_set) > 2:
                    for i in range(len(component_set)):
                        for j in range(i + 1, len(component_set)):
                            key = tuple(sorted([component_set[i], component_set[j]]))
                            if key not in rules:
                                rules[key] = result_id
        
        return rules


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='使用LLM构建知识图谱 - Mooper版本')
    parser.add_argument('--topics', type=str, default='data/xlsx/topics.csv',
                        help='知识点CSV文件路径')
    parser.add_argument('--output', type=str, default='data/knowledge_graph.json',
                        help='输出知识图谱文件路径')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='每批处理的知识点数量')
    parser.add_argument('--resume', action='store_true',
                        help='继续之前未完成的构建')
    
    args = parser.parse_args()
    
    builder = KnowledgeGraphBuilder(
        topics_file=args.topics,
        output_path=args.output,
        batch_size=args.batch_size
    )
    
    if args.resume:
        builder.load_existing_graph()
    
    graph = builder.build_graph_by_domain()
    
    print("\n" + "="*60)
    print("知识图谱构建统计")
    print("="*60)
    print(f"领域数: {len(graph.get('domains', {}))}")
    print(f"组合关系数: {len(graph.get('composites', {}))}")
    
    rules = builder.get_composite_rules()
    print(f"可用于R2的组合规则数: {len(rules)}")
    print("="*60)


if __name__ == '__main__':
    main()
