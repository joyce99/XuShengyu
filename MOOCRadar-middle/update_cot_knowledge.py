# -*- coding: utf-8 -*-
"""
使用CoT隐式知识点更新MOOCRadar数据集的knowledge_code字段
"""

import pandas as pd
import json
import os
import re


def parse_implicit_knowledge_ids(knowledge_ids):
    """解析implicit_knowledge_ids字段，返回整数数组"""
    if pd.isna(knowledge_ids) or knowledge_ids == '' or knowledge_ids == '[]':
        return []

    try:
        if isinstance(knowledge_ids, str):
            # 处理字符串格式，如 "['101', '205']" 或 "[101, 205]"
            if knowledge_ids.startswith('[') and knowledge_ids.endswith(']'):
                try:
                    # 尝试用eval解析
                    knowledge_list = eval(knowledge_ids)
                    return [int(k) for k in knowledge_list]
                except:
                    # 用正则提取数字
                    numbers = re.findall(r'\d+', knowledge_ids)
                    return [int(n) for n in numbers]
            else:
                # 逗号分隔的格式
                parts = [p.strip().strip("'\"") for p in knowledge_ids.split(',') if p.strip()]
                return [int(p) for p in parts if p.isdigit()]

        elif isinstance(knowledge_ids, (list, tuple)):
            return [int(k) for k in knowledge_ids]

        else:
            val = str(knowledge_ids).strip()
            return [int(val)] if val.isdigit() else []

    except Exception as e:
        print(f"  警告: 解析implicit_knowledge_ids时出错: {knowledge_ids}, 错误: {e}")
        return []


def load_data_files(
    cot_result_file: str = "xlsx/cot_implicit_knowledge_t0_5.xlsx",
    mapping_file: str = "problem_id_mapping.json",
    train_file: str = "MOOCRadar-middle/train.json",
    val_file: str = "MOOCRadar-middle/val.json"
):
    """加载所有需要的数据文件"""
    print("正在加载数据文件...")

    try:
        # 1) 加载CoT预测结果 Excel
        cot_df = pd.read_excel(cot_result_file)
        print(f"已加载 {cot_result_file}，共 {len(cot_df)} 行")

        # 2) 加载 problem_id_mapping.json
        with open(mapping_file, 'r', encoding='utf-8') as f:
            problem_id_mapping = json.load(f)
        print(f"已加载 {mapping_file}，共 {len(problem_id_mapping)} 条映射")

        # 3) 加载 train.json / val.json
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"已加载 {train_file}，共 {len(train_data)} 条记录")

        with open(val_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        print(f"已加载 {val_file}，共 {len(val_data)} 条记录")

        return cot_df, problem_id_mapping, train_data, val_data

    except Exception as e:
        print(f"加载数据文件时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None, None, None


def update_json_data(train_data, val_data, cot_df, problem_id_mapping):
    """更新JSON数据中的knowledge_code字段"""
    print("\n正在更新JSON数据...")

    try:
        # 1. 构建 problem_id -> implicit_knowledge_ids 映射
        problem_to_knowledge = {}
        
        for _, row in cot_df.iterrows():
            problem_id = str(row.get('problem_id', '')).strip()
            if not problem_id or problem_id.lower() == 'nan':
                continue
            
            # 解析 implicit_knowledge_ids
            knowledge_ids = parse_implicit_knowledge_ids(row.get('implicit_knowledge_ids', []))
            problem_to_knowledge[problem_id] = knowledge_ids

        print(f"已创建 {len(problem_to_knowledge)} 个 problem_id -> implicit_knowledge_ids 映射")
        
        # 统计非空知识点的数量
        non_empty = sum(1 for v in problem_to_knowledge.values() if v)
        print(f"其中 {non_empty} 个有非空知识点")

        # 2. 构建 exer_id -> problem_id 映射
        exer_to_problem = {}
        for k, v in problem_id_mapping.items():
            try:
                exer_id = int(k)
                problem_id = str(v).strip()
                exer_to_problem[exer_id] = problem_id
            except Exception as e:
                print(f"  警告: 解析映射失败: {k}: {v}, 错误: {e}")

        print(f"已创建 {len(exer_to_problem)} 个 exer_id -> problem_id 映射")

        # 3. 更新函数
        def update_one_split(data, split_name):
            updated_count = 0
            empty_count = 0
            not_found_in_mapping = 0
            not_found_in_excel = 0

            for item in data:
                if 'exer_id' not in item:
                    continue
                    
                exer_id = int(item['exer_id'])

                if exer_id in exer_to_problem:
                    problem_id = exer_to_problem[exer_id]

                    if problem_id in problem_to_knowledge:
                        knowledge_ids = problem_to_knowledge[problem_id]
                        item['knowledge_code'] = knowledge_ids

                        if knowledge_ids:
                            updated_count += 1
                        else:
                            empty_count += 1
                    else:
                        not_found_in_excel += 1
                        if not_found_in_excel <= 3:
                            print(f"[{split_name}] Excel中未找到: exer_id={exer_id}, problem_id={problem_id}")
                else:
                    not_found_in_mapping += 1
                    if not_found_in_mapping <= 3:
                        print(f"[{split_name}] mapping中未找到: exer_id={exer_id}")

            print(f"\n{split_name}.json 更新统计:")
            print(f"  - 成功更新(非空): {updated_count}")
            print(f"  - 设为空列表: {empty_count}")
            print(f"  - mapping中未找到: {not_found_in_mapping}")
            print(f"  - Excel中未找到: {not_found_in_excel}")

            return data

        updated_train = update_one_split(train_data, "train")
        updated_val = update_one_split(val_data, "val")

        return updated_train, updated_val

    except Exception as e:
        print(f"更新JSON数据时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None


def save_json_data(updated_train, updated_val, output_dir="MOOCRadar-middle", suffix="_cot"):
    """保存更新后的JSON数据"""
    print("\n正在保存更新后的JSON数据...")

    try:
        train_output = os.path.join(output_dir, f"train{suffix}.json")
        with open(train_output, 'w', encoding='utf-8') as f:
            json.dump(updated_train, f, ensure_ascii=False, indent=4)
        print(f"已保存: {train_output}")

        val_output = os.path.join(output_dir, f"val{suffix}.json")
        with open(val_output, 'w', encoding='utf-8') as f:
            json.dump(updated_val, f, ensure_ascii=False, indent=4)
        print(f"已保存: {val_output}")

        return True

    except Exception as e:
        print(f"保存JSON数据时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='使用CoT隐式知识点更新MOOCRadar数据集')
    parser.add_argument('--cot', type=str, 
                        default="xlsx/cot_implicit_knowledge_t0_5.xlsx",
                        help='CoT预测结果文件')
    parser.add_argument('--mapping', type=str,
                        default="data/problem_id_mapping.json",
                        help='problem_id映射文件')
    parser.add_argument('--train', type=str,
                        default="data/MOOCRadar-middle/train.json",
                        help='原始train.json文件')
    parser.add_argument('--val', type=str,
                        default="data/MOOCRadar-middle/val.json",
                        help='原始val.json文件')
    parser.add_argument('--output-dir', type=str,
                        default="data/MOOCRadar-middle",
                        help='输出目录')
    parser.add_argument('--suffix', type=str,
                        default="_cot",
                        help='输出文件后缀')
    
    args = parser.parse_args()

    print("=" * 60)
    print("使用CoT隐式知识点更新MOOCRadar数据集")
    print("=" * 60)

    # 加载数据
    cot_df, problem_id_mapping, train_data, val_data = load_data_files(
        cot_result_file=args.cot,
        mapping_file=args.mapping,
        train_file=args.train,
        val_file=args.val
    )
    
    if cot_df is None:
        print("加载数据失败，程序终止")
        return

    # 更新数据
    updated_train, updated_val = update_json_data(
        train_data, val_data, cot_df, problem_id_mapping
    )
    
    if updated_train is None:
        print("更新数据失败，程序终止")
        return

    # 保存数据
    success = save_json_data(
        updated_train, updated_val,
        output_dir=args.output_dir,
        suffix=args.suffix
    )
    
    if success:
        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)


if __name__ == "__main__":
    main()
