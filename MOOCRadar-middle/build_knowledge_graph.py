"""
知识图谱构建模块 (Knowledge Graph Builder)

使用 LLM 为知识点构建层级关系和组合关系图谱。

两阶段构建策略：
1. 阶段1：对所有知识点进行领域分类
2. 阶段2：在同一领域内分析组合关系（确保相关知识点能被一起分析）

输出结构：
{
    "hierarchy": {
        "父知识点ID": ["子知识点ID1", "子知识点ID2", ...]
    },
    "composites": {
        "组合知识点ID": {
            "name": "组合知识点名称",
            "component_sets": [["子知识点ID1", "子知识点ID2"], ...]
        }
    },
    "domains": {
        "领域名称": ["知识点ID1", "知识点ID2", ...]
    }
}
"""

import os
import json
import time
from typing import List, Dict, Set, Tuple, Optional
from openai import OpenAI
from tqdm import tqdm
from itertools import combinations
import config

class KnowledgeGraphBuilder:
    """基于LLM的知识图谱构建器"""
    
    def __init__(self, concept_mapping_path: str = "concept_mapping.json",
                 output_path: str = "knowledge_graph.json",
                 batch_size: int = 30,
                 api_key: str = None,
                 base_url: str = None):
        """
        初始化构建器
        
        Args:
            concept_mapping_path: 知识点映射文件路径
            output_path: 输出知识图谱文件路径
            batch_size: 每次LLM调用处理的知识点数量
            api_key: LLM API密钥
            base_url: LLM API地址
        """
        self.concept_mapping_path = concept_mapping_path
        self.output_path = output_path
        self.batch_size = batch_size
        
        # 加载知识点
        with open(concept_mapping_path, 'r', encoding='utf-8') as f:
            self.concept_mapping = json.load(f)
        
        # 构建反向映射
        self.reverse_mapping = {v: k for k, v in self.concept_mapping.items()}
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key or os.environ.get("DASHSCOPE_API_KEY"),
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 知识图谱结构
        self.knowledge_graph = {
            "hierarchy": {},      # 父子关系: {父ID: [子ID列表]}
            "composites": {},     # 组合关系: {组合ID: {name, component_sets}}
            "domains": {},        # 领域分类: {领域名: [知识点ID列表]}
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
                        {"role": "system", "content": "你是一个知识图谱专家，擅长分析知识点之间的层级关系和组合关系。请严格按照要求的JSON格式输出。"},
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
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取JSON块
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试找到 { 和 } 之间的内容
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return {}
    
    def analyze_batch_relationships(self, concept_names: List[str]) -> Dict:
        """
        分析一批知识点的关系
        
        Args:
            concept_names: 知识点名称列表
            
        Returns:
            包含关系的字典
        """
        concepts_str = "\n".join([f"- {name}" for name in concept_names])
        
        prompt = f"""分析以下知识点之间的关系，找出：
1. 层级关系（哪些是父概念，哪些是子概念）
2. 组合关系（哪些知识点可以组合成另一个知识点）
3. 同级关系（哪些知识点是同一层级的并列概念）

知识点列表：
{concepts_str}

请以JSON格式输出，格式如下：
{{
    "hierarchy": [
        {{"parent": "父知识点名称", "children": ["子知识点1", "子知识点2"]}}
    ],
    "composites": [
        {{"result": "组合后的知识点", "components": ["组成部分1", "组成部分2"]}}
    ],
    "siblings": [
        ["同级知识点1", "同级知识点2", "同级知识点3"]
    ]
}}

注意：
1. 只输出在给定列表中存在的知识点
2. 组合关系举例：如果"加法"和"减法"可以组合成"四则运算"，则记录
3. 如果没有发现某类关系，对应数组为空
4. 只输出JSON，不要其他内容"""

        response = self._call_llm(prompt)
        return self._parse_llm_json(response)
    
    def analyze_composite_relationships(self, concept_names: List[str], 
                                         all_concepts: List[str]) -> Dict:
        """
        专门分析组合关系：给定一批知识点，找出它们可以组合成哪些其他知识点
        
        Args:
            concept_names: 要分析的知识点列表
            all_concepts: 所有知识点列表（用于查找组合目标）
            
        Returns:
            组合关系字典
        """
        concepts_str = "\n".join([f"- {name}" for name in concept_names])
        
        prompt = f"""你是知识图谱专家。分析以下知识点，找出哪些知识点可以组合成更高层级的概念。

待分析的知识点：
{concepts_str}

请找出这些知识点中，哪些可以两两组合（或多个组合）形成一个更通用/更高层级的知识点。

组合关系的例子：
- "加法" + "减法" → "加减运算" 或 "四则运算"
- "直接推理" + "间接推理" → "推理"
- "民事权利" + "民事义务" → "民事法律关系"

请以JSON格式输出：
{{
    "composites": [
        {{
            "components": ["知识点A", "知识点B"],
            "result": "组合后的知识点",
            "confidence": 0.9,
            "reason": "组合原因简述"
        }}
    ]
}}

注意：
1. components 中的知识点必须来自上面的待分析列表
2. result 是组合后的更高层级概念（可以不在列表中）
3. confidence 是你对这个组合关系的置信度 (0-1)
4. 只输出有意义的组合，不要强行组合无关的知识点
5. 只输出JSON"""

        response = self._call_llm(prompt)
        return self._parse_llm_json(response)
    
    def build_graph_by_domain(self):
        """
        【推荐】按领域分批构建知识图谱（两阶段方法）
        
        阶段1: 对所有知识点进行领域分类
        阶段2: 在同一领域内的所有知识点之间寻找组合关系
        
        这样可以确保同一领域的相关知识点能被一起分析，不会因为分批而遗漏关系。
        """
        all_concepts = list(self.concept_mapping.values())
        
        # ========== 阶段1：领域分类 ==========
        print("="*60)
        print("阶段1: 对知识点进行领域分类...")
        print("="*60)
        
        domains = self._classify_domains(all_concepts)
        
        # 存储领域分类结果
        for domain, concepts in domains.items():
            concept_ids = [self.reverse_mapping.get(c) for c in concepts if c in self.reverse_mapping]
            concept_ids = [cid for cid in concept_ids if cid]
            if concept_ids:
                self.knowledge_graph["domains"][domain] = concept_ids
        
        print(f"\n分类完成！共 {len(domains)} 个领域:")
        for domain, concepts in sorted(domains.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  - {domain}: {len(concepts)} 个知识点")
        
        # ========== 阶段2：领域内分析组合关系 ==========
        print("\n" + "="*60)
        print("阶段2: 在各领域内分析组合关系...")
        print("="*60)
        
        for domain, concepts in tqdm(domains.items(), desc="处理领域"):
            if len(concepts) < 2:
                continue
            
            # 对于较大的领域，需要分批但使用滑动窗口
            if len(concepts) <= self.batch_size:
                # 小领域：一次性分析所有知识点
                self._analyze_domain_composites(concepts, domain)
            else:
                # 大领域：使用滑动窗口分批，确保相邻批次有重叠
                self._analyze_large_domain_composites(concepts, domain)
            
            time.sleep(0.3)  # 避免API限流
        
        # 保存结果
        self._save_graph()
        return self.knowledge_graph
    
    def _analyze_domain_composites(self, concepts: List[str], domain: str):
        """分析单个领域内的组合关系"""
        result = self.analyze_composite_relationships_in_domain(concepts, domain)
        
        if result and "composites" in result:
            for comp in result["composites"]:
                self._store_composite_relationship(comp)
    
    def _analyze_large_domain_composites(self, concepts: List[str], domain: str):
        """
        对大领域使用智能分批策略
        
        策略：不是简单分批，而是让LLM先识别该领域内的子主题，
        然后在每个子主题内分析组合关系
        """
        # 先让LLM对领域内的知识点进行子主题分组
        sub_groups = self._get_sub_groups(concepts, domain)
        
        for sub_topic, sub_concepts in sub_groups.items():
            if len(sub_concepts) < 2:
                continue
            
            if len(sub_concepts) <= self.batch_size:
                self._analyze_domain_composites(sub_concepts, f"{domain}/{sub_topic}")
            else:
                # 如果子主题还是太大，使用滑动窗口
                overlap = self.batch_size // 3  # 1/3 重叠
                for i in range(0, len(sub_concepts), self.batch_size - overlap):
                    batch = sub_concepts[i:i + self.batch_size]
                    if len(batch) >= 2:
                        self._analyze_domain_composites(batch, f"{domain}/{sub_topic}")
                    time.sleep(0.3)
    
    def _get_sub_groups(self, concepts: List[str], domain: str) -> Dict[str, List[str]]:
        """让LLM对领域内的知识点进行子主题分组"""
        concepts_str = "\n".join([f"- {c}" for c in concepts])
        
        prompt = f"""将以下"{domain}"领域的知识点按子主题分组：

{concepts_str}

请将这些知识点按照更细分的子主题分组，每组应包含语义相关的知识点。
相关的知识点应该分在同一组，这样便于后续分析它们之间的组合关系。

以JSON格式输出：
{{
    "子主题1": ["知识点1", "知识点2"],
    "子主题2": ["知识点3", "知识点4"]
}}

只输出JSON。"""
        
        response = self._call_llm(prompt)
        result = self._parse_llm_json(response)
        
        if not result:
            # 如果解析失败，按原样返回
            return {domain: concepts}
        
        return result
    
    def analyze_composite_relationships_in_domain(self, concept_names: List[str], 
                                                    domain: str) -> Dict:
        """
        分析同一领域内知识点的组合关系
        
        Args:
            concept_names: 同一领域的知识点列表
            domain: 领域名称
            
        Returns:
            组合关系字典
        """
        concepts_str = "\n".join([f"- {name}" for name in concept_names])
        
        prompt = f"""你是知识图谱专家。以下是"{domain}"领域的知识点，请分析它们之间的组合关系。

知识点列表：
{concepts_str}

请找出这些知识点中，哪些可以两两组合（或多个组合）形成一个更通用/更高层级的知识点。

组合关系的例子：
- "加法" + "减法" → "四则运算"
- "直接推理" + "间接推理" → "推理"  
- "民事权利" + "民事义务" → "民事法律关系"
- "联言命题" + "选言命题" + "假言命题" → "复合命题"

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

    def build_graph_simple(self):
        """
        简单构建：直接分批分析所有知识点的组合关系
        """
        all_concepts = list(self.concept_mapping.values())
        total_batches = (len(all_concepts) + self.batch_size - 1) // self.batch_size
        
        print(f"开始构建知识图谱，共 {len(all_concepts)} 个知识点，分 {total_batches} 批处理...")
        
        for i in tqdm(range(0, len(all_concepts), self.batch_size), desc="处理进度"):
            batch = all_concepts[i:i + self.batch_size]
            
            # 分析组合关系
            result = self.analyze_composite_relationships(batch, all_concepts)
            
            if result and "composites" in result:
                for comp in result["composites"]:
                    self._store_composite_relationship(comp)
            
            time.sleep(0.5)  # 避免API限流
            
            # 每处理20批保存一次
            if (i // self.batch_size + 1) % 20 == 0:
                self._save_graph()
        
        self._save_graph()
        print(f"\n知识图谱构建完成！共发现 {len(self.knowledge_graph['composites'])} 个组合关系")
        return self.knowledge_graph
    
    def _classify_domains(self, concepts: List[str]) -> Dict[str, List[str]]:
        """使用LLM对知识点进行领域分类"""
        domains = {}
        
        # 分批处理
        batch_size = 50
        for i in tqdm(range(0, len(concepts), batch_size), desc="领域分类"):
            batch = concepts[i:i + batch_size]
            concepts_str = "\n".join([f"- {c}" for c in batch])
            
            prompt = f"""将以下知识点按学科/领域分类：

{concepts_str}

请以JSON格式输出：
{{
    "领域名称1": ["知识点1", "知识点2"],
    "领域名称2": ["知识点3", "知识点4"]
}}

领域示例：数学、物理、化学、生物、计算机、法律、经济、医学、逻辑学等。
只输出JSON。"""
            
            response = self._call_llm(prompt)
            result = self._parse_llm_json(response)
            
            for domain, domain_concepts in result.items():
                if domain not in domains:
                    domains[domain] = []
                domains[domain].extend(domain_concepts)
            
            time.sleep(0.3)
        
        return domains
    
    def _analyze_and_store_relationships(self, concepts: List[str]):
        """分析并存储一批知识点的关系"""
        result = self.analyze_batch_relationships(concepts)
        
        if not result:
            return
        
        # 存储层级关系
        if "hierarchy" in result:
            for rel in result["hierarchy"]:
                parent = rel.get("parent", "")
                children = rel.get("children", [])
                if parent and children:
                    parent_id = self.reverse_mapping.get(parent)
                    if parent_id:
                        if parent_id not in self.knowledge_graph["hierarchy"]:
                            self.knowledge_graph["hierarchy"][parent_id] = []
                        for child in children:
                            child_id = self.reverse_mapping.get(child)
                            if child_id and child_id not in self.knowledge_graph["hierarchy"][parent_id]:
                                self.knowledge_graph["hierarchy"][parent_id].append(child_id)
        
        # 存储组合关系
        if "composites" in result:
            for comp in result["composites"]:
                self._store_composite_relationship(comp)
        
        # 存储同级关系
        if "siblings" in result:
            for sibling_group in result["siblings"]:
                sibling_ids = [self.reverse_mapping.get(s) for s in sibling_group if s in self.reverse_mapping]
                sibling_ids = [s for s in sibling_ids if s]
                for sid in sibling_ids:
                    if sid not in self.knowledge_graph["siblings"]:
                        self.knowledge_graph["siblings"][sid] = []
                    for other_sid in sibling_ids:
                        if other_sid != sid and other_sid not in self.knowledge_graph["siblings"][sid]:
                            self.knowledge_graph["siblings"][sid].append(other_sid)
    
    def _store_composite_relationship(self, comp: Dict):
        """存储单个组合关系"""
        components = comp.get("components", [])
        result_name = comp.get("result", "")
        
        if len(components) < 2 or not result_name:
            return
        
        # 获取组合结果的ID（如果存在于知识点库中）
        result_id = self.reverse_mapping.get(result_name)
        
        # 获取组成部分的ID
        component_ids = []
        for c in components:
            cid = self.reverse_mapping.get(c)
            if cid:
                component_ids.append(cid)
        
        if len(component_ids) < 2:
            return
        
        # 如果组合结果不在知识点库中，用组件名称作为键
        if not result_id:
            result_id = f"_composite_{result_name}"
        
        if result_id not in self.knowledge_graph["composites"]:
            self.knowledge_graph["composites"][result_id] = {
                "name": result_name,
                "component_sets": []
            }
        
        # 存储组件组合（排序以确保一致性）
        sorted_components = sorted(component_ids)
        if sorted_components not in self.knowledge_graph["composites"][result_id]["component_sets"]:
            self.knowledge_graph["composites"][result_id]["component_sets"].append(sorted_components)
    
    def _save_graph(self):
        """保存知识图谱到文件"""
        import datetime
        self.knowledge_graph["metadata"]["build_time"] = datetime.datetime.now().isoformat()
        
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
        """
        将知识图谱转换为组合规则表，供 R2 规则使用
        
        Returns:
            {(component1_id, component2_id): result_id}
        """
        rules = {}
        
        for result_id, data in self.knowledge_graph.get("composites", {}).items():
            # 跳过不在知识点库中的组合结果
            if result_id.startswith("_composite_"):
                continue
            
            for component_set in data.get("component_sets", []):
                if len(component_set) == 2:
                    # 二元组合
                    key = tuple(sorted(component_set))
                    rules[key] = result_id
                elif len(component_set) > 2:
                    # 多元组合：生成所有二元子组合
                    for i in range(len(component_set)):
                        for j in range(i + 1, len(component_set)):
                            key = tuple(sorted([component_set[i], component_set[j]]))
                            if key not in rules:
                                rules[key] = result_id
        
        return rules


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用LLM构建知识图谱')
    parser.add_argument('--concept-mapping', type=str, default='concept_mapping.json',
                        help='知识点映射文件路径')
    parser.add_argument('--output', type=str, default='knowledge_graph.json',
                        help='输出知识图谱文件路径')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='每批处理的知识点数量')
    parser.add_argument('--mode', type=str, choices=['simple', 'domain'], default='domain',
                        help='构建模式: simple=简单分批(不推荐), domain=按领域分类后处理(推荐)')
    parser.add_argument('--resume', action='store_true',
                        help='继续之前未完成的构建')
    
    args = parser.parse_args()
    
    builder = KnowledgeGraphBuilder(
        concept_mapping_path=args.concept_mapping,
        output_path=args.output,
        batch_size=args.batch_size
    )
    
    if args.resume:
        builder.load_existing_graph()
    
    if args.mode == 'simple':
        print("⚠️ 警告：simple模式会导致跨批次的知识点无法建立关系，推荐使用 --mode domain")
        graph = builder.build_graph_simple()
    else:
        graph = builder.build_graph_by_domain()
    
    # 输出统计信息
    print("\n" + "="*60)
    print("知识图谱构建统计")
    print("="*60)
    print(f"领域数: {len(graph.get('domains', {}))}")
    print(f"层级关系数: {len(graph.get('hierarchy', {}))}")
    print(f"组合关系数: {len(graph.get('composites', {}))}")
    
    # 生成组合规则表
    rules = builder.get_composite_rules()
    print(f"可用于R2的组合规则数: {len(rules)}")
    print("="*60)


if __name__ == '__main__':
    main()