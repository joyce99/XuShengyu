# -*- coding: utf-8 -*-
"""
基于规则的Q矩阵增强模块 - Mooper数据集版本

实现两个融合规则：
1. 保留规则 R1: 显式 ∪ 隐式 → 增强
2. 补充规则 R2: 如果 k_a + k_b → k_c，则也加入 k_c
"""

import pandas as pd
import json
import ast
from typing import List, Dict, Set
import os


class RuleBasedQMatrixEnhancer:
    """基于规则的Q矩阵增强器"""
    
    def __init__(self, topics_file: str = None, 
                 knowledge_graph_path: str = None):
        """
        初始化增强器
        
        Args:
            topics_file: 知识点CSV文件路径
            knowledge_graph_path: 知识图谱文件路径
        """
        self.concept_mapping = {}
        self.reverse_mapping = {}
        self.composite_rules = {}
        
        if topics_file and os.path.exists(topics_file):
            self._load_topics(topics_file)
        
        if knowledge_graph_path and os.path.exists(knowledge_graph_path):
            self._load_rules_from_knowledge_graph(knowledge_graph_path)
        else:
            self._build_default_composite_rules()
    
    def _load_topics(self, path: str):
        """加载知识点CSV文件"""
        topics_df = pd.read_csv(path)
        self.concept_mapping = dict(zip(
            topics_df.topic_id.astype(str),
            topics_df.topic_name
        ))
        self.reverse_mapping = {v: k for k, v in self.concept_mapping.items()}
        print(f"已加载 {len(self.concept_mapping)} 个知识点映射")
    
    def _load_rules_from_knowledge_graph(self, path: str):
        """从知识图谱文件加载组合规则"""
        with open(path, 'r', encoding='utf-8') as f:
            knowledge_graph = json.load(f)
        
        composites = knowledge_graph.get("composites", {})
        
        for result_id, data in composites.items():
            if result_id.startswith("_composite_"):
                continue
            
            for component_set in data.get("component_sets", []):
                if len(component_set) == 2:
                    key = tuple(sorted(component_set))
                    self.composite_rules[key] = result_id
                elif len(component_set) > 2:
                    for i in range(len(component_set)):
                        for j in range(i + 1, len(component_set)):
                            key = tuple(sorted([component_set[i], component_set[j]]))
                            if key not in self.composite_rules:
                                self.composite_rules[key] = result_id
        
        print(f"从知识图谱加载了 {len(self.composite_rules)} 条组合规则")
    
    def _build_default_composite_rules(self):
        """基于知识点名称构建默认的组合规则"""
        if not self.concept_mapping:
            return
        
        # 编程领域的预定义组合模式
        predefined_composites = {
            "循环语句": ["for循环", "while循环", "do-while"],
            "条件语句": ["if语句", "switch语句", "条件判断"],
            "控制语句": ["循环语句", "条件语句", "跳转语句"],
            "面向对象": ["类", "对象", "继承", "多态", "封装", "接口"],
            "数据结构": ["数组", "链表", "栈", "队列", "树", "图"],
            "基本运算": ["加法", "减法", "乘法", "除法"],
            "文件操作": ["读文件", "写文件", "文件流"],
        }
        
        for composite_name, sub_keywords in predefined_composites.items():
            composite_id = self.reverse_mapping.get(composite_name)
            if not composite_id:
                continue
            
            sub_ids = []
            for keyword in sub_keywords:
                for kname, kid in self.reverse_mapping.items():
                    if keyword in kname or kname in keyword:
                        sub_ids.append(kid)
            
            for i in range(len(sub_ids)):
                for j in range(i + 1, len(sub_ids)):
                    rule_key = tuple(sorted([sub_ids[i], sub_ids[j]]))
                    self.composite_rules[rule_key] = composite_id
        
        print(f"已构建 {len(self.composite_rules)} 条默认组合规则")

    def _parse_knowledge_ids(self, value) -> List[str]:
        """解析知识点ID列表"""
        if pd.isna(value) or value is None:
            return []
        
        if isinstance(value, list):
            return [str(x) for x in value]
        
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (ValueError, SyntaxError):
                pass
            
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except json.JSONDecodeError:
                pass
            
            if ',' in value:
                return [x.strip().strip("'\"") for x in value.split(',')]
            
            return [value]
        
        return [str(value)]
    
    def apply_retention_rule(self, explicit_kcs: List[str], 
                             implicit_kcs: List[str]) -> Set[str]:
        """R1: 保留规则 - 合并显式和隐式知识点"""
        return set(explicit_kcs) | set(implicit_kcs)
    
    def apply_supplementary_rule(self, explicit_kcs: List[str],
                                  implicit_kcs: List[str]) -> Set[str]:
        """R2: 补充规则 - 根据组合规则推导新的知识点"""
        derived_kcs = set()
        
        if not self.composite_rules:
            return derived_kcs
        
        # 显式 x 隐式
        for exp_kc in explicit_kcs:
            for imp_kc in implicit_kcs:
                rule_key = tuple(sorted([exp_kc, imp_kc]))
                if rule_key in self.composite_rules:
                    derived_kcs.add(self.composite_rules[rule_key])
        
        # 显式 x 显式
        for i, exp_kc1 in enumerate(explicit_kcs):
            for exp_kc2 in explicit_kcs[i+1:]:
                rule_key = tuple(sorted([exp_kc1, exp_kc2]))
                if rule_key in self.composite_rules:
                    derived_kcs.add(self.composite_rules[rule_key])
        
        # 隐式 x 隐式
        for i, imp_kc1 in enumerate(implicit_kcs):
            for imp_kc2 in implicit_kcs[i+1:]:
                rule_key = tuple(sorted([imp_kc1, imp_kc2]))
                if rule_key in self.composite_rules:
                    derived_kcs.add(self.composite_rules[rule_key])
        
        return derived_kcs
    
    def enhance_qmatrix(self, explicit_kcs: List[str], 
                        implicit_kcs: List[str]) -> Dict:
        """对单个练习应用完整的Q矩阵增强流程"""
        retained_kcs = self.apply_retention_rule(explicit_kcs, implicit_kcs)
        derived_kcs = self.apply_supplementary_rule(explicit_kcs, implicit_kcs)
        enhanced_kcs = retained_kcs | derived_kcs
        
        return {
            'explicit_kcs': list(explicit_kcs),
            'implicit_kcs': list(implicit_kcs),
            'retained_kcs': list(retained_kcs),
            'derived_kcs': list(derived_kcs),
            'enhanced_kcs': list(enhanced_kcs),
            'explicit_count': len(explicit_kcs),
            'implicit_count': len(implicit_kcs),
            'enhanced_count': len(enhanced_kcs),
            'derived_count': len(derived_kcs)
        }

    def process_files(self, explicit_file: str, implicit_file: str, 
                      output_file: str) -> pd.DataFrame:
        """
        处理两个Excel文件并生成增强后的Q矩阵
        
        Args:
            explicit_file: 显式知识点预测文件路径
            implicit_file: 隐式知识点预测文件路径
            output_file: 输出文件路径
        """
        print(f"正在读取显式知识点文件: {explicit_file}")
        explicit_df = pd.read_excel(explicit_file)
        
        print(f"正在读取隐式知识点文件: {implicit_file}")
        implicit_df = pd.read_excel(implicit_file)
        
        print(f"显式知识点数据: {len(explicit_df)} 条")
        print(f"隐式知识点数据: {len(implicit_df)} 条")
        
        results = []
        
        # 创建隐式数据的字典
        implicit_dict = {}
        for _, row in implicit_df.iterrows():
            cid = str(row.get('challenge_id', ''))
            if cid:
                implicit_dict[cid] = row
        
        processed_count = 0
        for _, exp_row in explicit_df.iterrows():
            challenge_id = str(exp_row.get('challenge_id', ''))
            
            # 获取显式知识点
            explicit_kcs = self._parse_knowledge_ids(exp_row.get('knowledge_ids', []))
            
            # 获取隐式知识点
            implicit_kcs = []
            if challenge_id in implicit_dict:
                imp_row = implicit_dict[challenge_id]
                implicit_kcs = self._parse_knowledge_ids(
                    imp_row.get('implicit_knowledge_ids', [])
                )
            
            # 应用Q矩阵增强
            enhancement_result = self.enhance_qmatrix(explicit_kcs, implicit_kcs)
            
            # 构建结果记录
            result = {
                'challenge_id': challenge_id,
                'exercise_name': exp_row.get('name', ''),
                'explicit_knowledge_ids': json.dumps(enhancement_result['explicit_kcs'], ensure_ascii=False),
                'implicit_knowledge_ids': json.dumps(enhancement_result['implicit_kcs'], ensure_ascii=False),
                'enhanced_knowledge_ids': json.dumps(enhancement_result['enhanced_kcs'], ensure_ascii=False),
                'derived_knowledge_ids': json.dumps(enhancement_result['derived_kcs'], ensure_ascii=False),
                'explicit_count': enhancement_result['explicit_count'],
                'implicit_count': enhancement_result['implicit_count'],
                'enhanced_count': enhancement_result['enhanced_count'],
                'derived_count': enhancement_result['derived_count'],
            }
            
            # 添加知识点名称
            if self.concept_mapping:
                result['explicit_knowledge_names'] = json.dumps(
                    [self.concept_mapping.get(kc, f'Unknown({kc})') for kc in enhancement_result['explicit_kcs']],
                    ensure_ascii=False
                )
                result['implicit_knowledge_names'] = json.dumps(
                    [self.concept_mapping.get(kc, f'Unknown({kc})') for kc in enhancement_result['implicit_kcs']],
                    ensure_ascii=False
                )
                result['enhanced_knowledge_names'] = json.dumps(
                    [self.concept_mapping.get(kc, f'Unknown({kc})') for kc in enhancement_result['enhanced_kcs']],
                    ensure_ascii=False
                )
            
            results.append(result)
            processed_count += 1
            
            if processed_count % 500 == 0:
                print(f"已处理 {processed_count} 条记录...")
        
        result_df = pd.DataFrame(results)
        
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        result_df.to_excel(output_file, index=False)
        print(f"增强结果已保存至: {output_file}")
        
        self._print_statistics(result_df)
        
        return result_df
    
    def _print_statistics(self, df: pd.DataFrame):
        """打印统计信息"""
        print("\n" + "="*60)
        print("Q矩阵增强统计信息")
        print("="*60)
        print(f"总记录数: {len(df)}")
        print(f"平均显式知识点数: {df['explicit_count'].mean():.2f}")
        print(f"平均隐式知识点数: {df['implicit_count'].mean():.2f}")
        print(f"平均增强后知识点数: {df['enhanced_count'].mean():.2f}")
        print(f"平均推导知识点数: {df['derived_count'].mean():.2f}")
        if df['explicit_count'].sum() > 0:
            print(f"知识点增长率: {(df['enhanced_count'].sum() / df['explicit_count'].sum() - 1) * 100:.2f}%")
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='基于规则的Q矩阵增强模块 - Mooper版本')
    parser.add_argument('--explicit-file', type=str, 
                        default='prediction_results_unlimited_threshold0_5.xlsx',
                        help='显式知识点预测文件路径')
    parser.add_argument('--implicit-file', type=str,
                        default='cot_implicit_knowledge_t0_6.xlsx',
                        help='隐式知识点预测文件路径')
    parser.add_argument('--output-file', type=str,
                        default='enhanced_qmatrix_results.xlsx',
                        help='输出文件路径')
    parser.add_argument('--topics', type=str,
                        default='data/xlsx/topics.csv',
                        help='知识点CSV文件路径')
    parser.add_argument('--knowledge-graph', type=str,
                        default='data/knowledge_graph.json',
                        help='知识图谱文件路径')
    
    args = parser.parse_args()
    
    enhancer = RuleBasedQMatrixEnhancer(
        topics_file=args.topics,
        knowledge_graph_path=args.knowledge_graph
    )
    
    result_df = enhancer.process_files(
        explicit_file=args.explicit_file,
        implicit_file=args.implicit_file,
        output_file=args.output_file
    )
    
    print("\n处理完成！")
    return result_df


if __name__ == '__main__':
    main()
