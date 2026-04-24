"""
基于规则的Q矩阵增强模块 (Rule-based Q-matrix Enhancement Module)

该模块实现论文中的两个融合规则：
1. 保留规则 R1: 当KC c_a 和 c_b 分别与练习 e_j 在显式和隐式Q矩阵中关联时，
   将增强后的Q矩阵中两者都保留。
   Π_exp[j,a] ∧ Π_imp[j,b] ⇒ Π_enh[j,a] = 1 ∧ Π_enh[j,b] = 1

2. 补充规则 R2: 如果KC k_a 和 k_b 可以共同推导出一个新的组合KC k_c，
   那么这个隐式的 k_c 也应该在Q矩阵中设为1。
   {Π_exp[j,a] ∧ Π_imp[j,b] ∧ (c_a ⊕ c_b → c_c)} ⇒ Π_enh[j,c] = 1

输入文件：
- xlsx-parameter/prediction_results_unlimited_threshold0_5_zp_0.xlsx: 显式知识点预测
- xlsx/cot_implicit_knowledge_t0_5.xlsx: 隐式知识点预测

输出文件：
- xlsx/enhanced_qmatrix_results.xlsx: 融合后的增强Q矩阵结果
"""

import pandas as pd
import json
import ast
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import os


class RuleBasedQMatrixEnhancer:
    """基于规则的Q矩阵增强器"""
    
    def __init__(self, concept_mapping_path: str = None, 
                 composite_rules_path: str = None,
                 knowledge_graph_path: str = None):
        """
        初始化增强器
        
        Args:
            concept_mapping_path: 知识点ID到名称的映射文件路径
            composite_rules_path: 知识点组合规则文件路径（可选，旧格式）
            knowledge_graph_path: 知识图谱文件路径（可选，新格式，优先使用）
        """
        self.concept_mapping = {}
        self.reverse_mapping = {}  # 名称到ID的反向映射
        self.composite_rules = {}  # 组合规则: {(k_a, k_b): k_c}
        
        if concept_mapping_path and os.path.exists(concept_mapping_path):
            self._load_concept_mapping(concept_mapping_path)
        
        # 优先从知识图谱加载组合规则
        if knowledge_graph_path and os.path.exists(knowledge_graph_path):
            self._load_rules_from_knowledge_graph(knowledge_graph_path)
        elif composite_rules_path and os.path.exists(composite_rules_path):
            self._load_composite_rules(composite_rules_path)
        else:
            # 如果没有提供规则文件，使用默认的基于名称推断的规则
            self._build_default_composite_rules()
    
    def _load_concept_mapping(self, path: str):
        """加载知识点ID到名称的映射"""
        with open(path, 'r', encoding='utf-8') as f:
            self.concept_mapping = json.load(f)
        # 构建反向映射
        self.reverse_mapping = {v: k for k, v in self.concept_mapping.items()}
        print(f"已加载 {len(self.concept_mapping)} 个知识点映射")
    
    def _load_composite_rules(self, path: str):
        """加载知识点组合规则"""
        with open(path, 'r', encoding='utf-8') as f:
            self.composite_rules = json.load(f)
        print(f"已加载 {len(self.composite_rules)} 条组合规则")
    
    def _load_rules_from_knowledge_graph(self, path: str):
        """从知识图谱文件加载组合规则"""
        with open(path, 'r', encoding='utf-8') as f:
            knowledge_graph = json.load(f)
        
        # 从知识图谱的 composites 字段提取规则
        composites = knowledge_graph.get("composites", {})
        
        for result_id, data in composites.items():
            # 跳过不在知识点库中的组合结果
            if result_id.startswith("_composite_"):
                continue
            
            for component_set in data.get("component_sets", []):
                if len(component_set) == 2:
                    # 二元组合
                    key = tuple(sorted(component_set))
                    self.composite_rules[key] = result_id
                elif len(component_set) > 2:
                    # 多元组合：生成所有二元子组合
                    for i in range(len(component_set)):
                        for j in range(i + 1, len(component_set)):
                            key = tuple(sorted([component_set[i], component_set[j]]))
                            if key not in self.composite_rules:
                                self.composite_rules[key] = result_id
        
        print(f"从知识图谱加载了 {len(self.composite_rules)} 条组合规则")
    
    def _build_default_composite_rules(self):
        """
        基于知识点名称构建默认的组合规则
        
        规则逻辑：
        1. 如果两个知识点名称都是某个更通用知识点名称的子串或相关概念，
           则推断它们可以组合成该通用知识点
        2. 基于常见的知识点层级关系（如：加法+减法→加减法→四则运算）
        """
        if not self.concept_mapping:
            return
        
        # 预定义一些常见的组合模式
        # 格式: {组合知识点名称: [可能的子知识点关键词]}
        predefined_composites = {
            # 数学相关
            "四则运算": ["加法", "减法", "乘法", "除法", "加减", "乘除"],
            "加减法": ["加法", "减法"],
            "乘除法": ["乘法", "除法"],
            # 逻辑相关
            "复合命题": ["联言命题", "选言命题", "假言命题", "条件命题"],
            "命题逻辑": ["命题", "逻辑", "推理"],
            "逻辑推理": ["演绎推理", "归纳推理", "类比推理"],
            # 法律相关
            "民事权利": ["人身权", "财产权", "债权", "物权"],
            "民事义务": ["给付义务", "不作为义务"],
            # 编程相关
            "循环结构": ["for循环", "while循环", "循环语句"],
            "条件语句": ["if语句", "switch语句", "条件判断"],
            "数据结构": ["数组", "链表", "栈", "队列", "树", "图"],
        }
        
        # 根据预定义规则和实际知识点构建组合规则
        for composite_name, sub_keywords in predefined_composites.items():
            # 查找是否存在该组合知识点
            composite_id = self.reverse_mapping.get(composite_name)
            if not composite_id:
                continue
            
            # 查找可能的子知识点
            sub_ids = []
            for keyword in sub_keywords:
                for kname, kid in self.reverse_mapping.items():
                    if keyword in kname or kname in keyword:
                        sub_ids.append(kid)
            
            # 为所有子知识点对创建组合规则
            for i in range(len(sub_ids)):
                for j in range(i + 1, len(sub_ids)):
                    rule_key = tuple(sorted([sub_ids[i], sub_ids[j]]))
                    self.composite_rules[rule_key] = composite_id
        
        # 基于名称相似性自动推断组合规则
        self._infer_composite_rules_by_name()
        
        print(f"已构建 {len(self.composite_rules)} 条默认组合规则")
    
    def _infer_composite_rules_by_name(self):
        """基于知识点名称相似性推断组合规则"""
        if not self.concept_mapping:
            return
        
        # 找出可能是"组合"类型的知识点（名称较长或包含多个概念）
        for kid, kname in self.concept_mapping.items():
            # 检查是否有其他知识点是当前知识点名称的子串
            potential_subs = []
            for sub_id, sub_name in self.concept_mapping.items():
                if sub_id != kid and len(sub_name) >= 2:
                    # 检查子知识点名称是否是当前知识点名称的一部分
                    if sub_name in kname and len(sub_name) < len(kname):
                        potential_subs.append(sub_id)
            
            # 如果找到至少2个潜在的子知识点，创建组合规则
            if len(potential_subs) >= 2:
                for i in range(len(potential_subs)):
                    for j in range(i + 1, len(potential_subs)):
                        rule_key = tuple(sorted([potential_subs[i], potential_subs[j]]))
                        if rule_key not in self.composite_rules:
                            self.composite_rules[rule_key] = kid
    
    def _parse_knowledge_ids(self, value) -> List[str]:
        """解析知识点ID列表"""
        if pd.isna(value) or value is None:
            return []
        
        if isinstance(value, list):
            return [str(x) for x in value]
        
        if isinstance(value, str):
            try:
                # 尝试解析为Python列表
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (ValueError, SyntaxError):
                pass
            
            # 尝试解析为JSON
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except json.JSONDecodeError:
                pass
            
            # 尝试以逗号分隔
            if ',' in value:
                return [x.strip().strip("'\"") for x in value.split(',')]
            
            return [value]
        
        return [str(value)]
    
    def apply_retention_rule(self, explicit_kcs: List[str], 
                             implicit_kcs: List[str]) -> Set[str]:
        """
        应用保留规则 R1: 合并显式和隐式知识点
        
        Args:
            explicit_kcs: 显式Q矩阵中的知识点ID列表
            implicit_kcs: 隐式Q矩阵中的知识点ID列表
            
        Returns:
            合并后的知识点ID集合
        """
        # 简单地取并集
        enhanced_kcs = set(explicit_kcs) | set(implicit_kcs)
        return enhanced_kcs
    
    def apply_supplementary_rule(self, explicit_kcs: List[str],
                                  implicit_kcs: List[str]) -> Set[str]:
        """
        应用补充规则 R2: 根据组合规则推导新的隐式知识点
        
        如果KC k_a 和 k_b 分别在显式和隐式Q矩阵中，
        且它们可以组合成 k_c，则将 k_c 也加入增强后的Q矩阵
        
        Args:
            explicit_kcs: 显式Q矩阵中的知识点ID列表
            implicit_kcs: 隐式Q矩阵中的知识点ID列表
            
        Returns:
            通过组合规则推导出的新知识点ID集合
        """
        derived_kcs = set()
        
        if not self.composite_rules:
            return derived_kcs
        
        # 对于每一对 (显式KC, 隐式KC)，检查是否存在组合规则
        for exp_kc in explicit_kcs:
            for imp_kc in implicit_kcs:
                # 创建规则键（排序以确保一致性）
                rule_key = tuple(sorted([exp_kc, imp_kc]))
                
                if rule_key in self.composite_rules:
                    derived_kc = self.composite_rules[rule_key]
                    derived_kcs.add(derived_kc)
        
        # 同时检查显式KC之间的组合
        for i, exp_kc1 in enumerate(explicit_kcs):
            for exp_kc2 in explicit_kcs[i+1:]:
                rule_key = tuple(sorted([exp_kc1, exp_kc2]))
                if rule_key in self.composite_rules:
                    derived_kcs.add(self.composite_rules[rule_key])
        
        # 同时检查隐式KC之间的组合
        for i, imp_kc1 in enumerate(implicit_kcs):
            for imp_kc2 in implicit_kcs[i+1:]:
                rule_key = tuple(sorted([imp_kc1, imp_kc2]))
                if rule_key in self.composite_rules:
                    derived_kcs.add(self.composite_rules[rule_key])
        
        return derived_kcs
    
    def enhance_qmatrix(self, explicit_kcs: List[str], 
                        implicit_kcs: List[str]) -> Dict:
        """
        对单个练习应用完整的Q矩阵增强流程
        
        Args:
            explicit_kcs: 显式知识点ID列表
            implicit_kcs: 隐式知识点ID列表
            
        Returns:
            包含增强结果的字典
        """
        # R1: 保留规则 - 合并显式和隐式
        retained_kcs = self.apply_retention_rule(explicit_kcs, implicit_kcs)
        
        # R2: 补充规则 - 推导组合知识点
        derived_kcs = self.apply_supplementary_rule(explicit_kcs, implicit_kcs)
        
        # 最终增强后的知识点集合
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
                      output_file: str, 
                      explicit_threshold: float = 0.5,
                      implicit_threshold: float = 0.5) -> pd.DataFrame:
        """
        处理两个Excel文件并生成增强后的Q矩阵
        
        Args:
            explicit_file: 显式知识点预测文件路径
            implicit_file: 隐式知识点预测文件路径
            output_file: 输出文件路径
            explicit_threshold: 显式知识点的筛选阈值
            implicit_threshold: 隐式知识点的筛选阈值
            
        Returns:
            增强后的结果DataFrame
        """
        print(f"正在读取显式知识点文件: {explicit_file}")
        explicit_df = pd.read_excel(explicit_file)
        
        print(f"正在读取隐式知识点文件: {implicit_file}")
        implicit_df = pd.read_excel(implicit_file)
        
        print(f"显式知识点数据: {len(explicit_df)} 条")
        print(f"隐式知识点数据: {len(implicit_df)} 条")
        
        # 以problem_id为键合并数据
        results = []
        
        # 创建隐式数据的字典以便快速查找
        implicit_dict = {}
        for _, row in implicit_df.iterrows():
            pid = str(row.get('problem_id', ''))
            if pid:
                implicit_dict[pid] = row
        
        processed_count = 0
        for _, exp_row in explicit_df.iterrows():
            problem_id = str(exp_row.get('problem_id', ''))
            
            # 获取显式知识点
            explicit_kcs = self._parse_knowledge_ids(
                exp_row.get('knowledge_ids', [])
            )
            
            # 应用阈值筛选显式知识点
            if 'combined_scores' in exp_row and explicit_kcs:
                scores = self._parse_knowledge_ids(exp_row.get('combined_scores', []))
                if scores and len(scores) == len(explicit_kcs):
                    try:
                        filtered_explicit = [
                            kc for kc, score in zip(explicit_kcs, scores)
                            if float(score) >= explicit_threshold
                        ]
                        explicit_kcs = filtered_explicit if filtered_explicit else explicit_kcs[:5]
                    except (ValueError, TypeError):
                        pass
            
            # 获取隐式知识点
            implicit_kcs = []
            if problem_id in implicit_dict:
                imp_row = implicit_dict[problem_id]
                implicit_kcs = self._parse_knowledge_ids(
                    imp_row.get('implicit_knowledge_ids', [])
                )
            
            # 应用Q矩阵增强
            enhancement_result = self.enhance_qmatrix(explicit_kcs, implicit_kcs)
            
            # 构建结果记录
            result = {
                'problem_id': problem_id,
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
            
            # 添加知识点名称（如果有映射）
            if self.concept_mapping:
                explicit_names = [
                    self.concept_mapping.get(kc, f'Unknown({kc})') 
                    for kc in enhancement_result['explicit_kcs']
                ]
                implicit_names = [
                    self.concept_mapping.get(kc, f'Unknown({kc})') 
                    for kc in enhancement_result['implicit_kcs']
                ]
                enhanced_names = [
                    self.concept_mapping.get(kc, f'Unknown({kc})') 
                    for kc in enhancement_result['enhanced_kcs']
                ]
                derived_names = [
                    self.concept_mapping.get(kc, f'Unknown({kc})') 
                    for kc in enhancement_result['derived_kcs']
                ]
                
                result['explicit_knowledge_names'] = json.dumps(explicit_names, ensure_ascii=False)
                result['implicit_knowledge_names'] = json.dumps(implicit_names, ensure_ascii=False)
                result['enhanced_knowledge_names'] = json.dumps(enhanced_names, ensure_ascii=False)
                result['derived_knowledge_names'] = json.dumps(derived_names, ensure_ascii=False)
            
            results.append(result)
            processed_count += 1
            
            if processed_count % 500 == 0:
                print(f"已处理 {processed_count} 条记录...")
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(results)
        
        # 保存结果
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        result_df.to_excel(output_file, index=False)
        print(f"增强结果已保存至: {output_file}")
        
        # 打印统计信息
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
        print(f"知识点增长率: {(df['enhanced_count'].sum() / max(df['explicit_count'].sum(), 1) - 1) * 100:.2f}%")
        print("="*60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='基于规则的Q矩阵增强模块')
    parser.add_argument('--explicit-file', type=str, 
                        default='xlsx-parameter/prediction_results_unlimited_threshold0_5_zp_0.xlsx',
                        help='显式知识点预测文件路径')
    parser.add_argument('--implicit-file', type=str,
                        default='xlsx/cot_implicit_knowledge_t0_5.xlsx',
                        help='隐式知识点预测文件路径')
    parser.add_argument('--output-file', type=str,
                        default='xlsx/enhanced_qmatrix_results.xlsx',
                        help='输出文件路径')
    parser.add_argument('--concept-mapping', type=str,
                        default='concept_mapping.json',
                        help='知识点映射文件路径')
    parser.add_argument('--composite-rules', type=str,
                        default=None,
                        help='组合规则文件路径（可选，旧格式）')
    parser.add_argument('--knowledge-graph', type=str,
                        default='knowledge_graph.json',
                        help='知识图谱文件路径（优先使用）')
    parser.add_argument('--explicit-threshold', type=float,
                        default=0.5,
                        help='显式知识点筛选阈值')
    parser.add_argument('--implicit-threshold', type=float,
                        default=0.5,
                        help='隐式知识点筛选阈值')
    
    args = parser.parse_args()
    
    # 创建增强器
    enhancer = RuleBasedQMatrixEnhancer(
        concept_mapping_path=args.concept_mapping,
        composite_rules_path=args.composite_rules,
        knowledge_graph_path=args.knowledge_graph
    )
    
    # 处理文件
    result_df = enhancer.process_files(
        explicit_file=args.explicit_file,
        implicit_file=args.implicit_file,
        output_file=args.output_file,
        explicit_threshold=args.explicit_threshold,
        implicit_threshold=args.implicit_threshold
    )
    
    print("\n处理完成！")
    return result_df


if __name__ == '__main__':
    main()