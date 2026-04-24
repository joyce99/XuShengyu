# -*- coding: utf-8 -*-
"""
使用CoT隐式知识点更新Mooper数据集的knowledge_code字段
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
            if knowledge_ids.startswith('[') and knowledge_ids.endswith(']'):
                try:
                    knowledge_list = eval(knowledge_ids)
                    return [int(k) for k in knowledge_list]
                except:
                    numbers = re.findall(r'\d+', knowledge_ids)
                    return [int(n) for n in numbers]
            else:
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


def load_data_files(cot_result_file, mapping_file, train_file, val_file):
    """加载所有需要的数据文件"""
    print("正在加载数据文件...")

    try:
        # 1) 加载CoT预测结果 Excel
        cot_df = pd.read_excel(cot_result_file)
        print(f"已加载 {cot_result_file}，共 {len(cot_df)} 行")

        # 2) 加载 exercise_id_mapping.json (challenge_id -> new_id/exer_id)
        with open(mapping_file, 'r', encoding='utf-8') as f:
            id_mapping = json.load(f)
        print(f"已加载 {mapping_file}，共 {len(id_mapping)} 条映射")

        # 3) 加载 train.json / val.json
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"已加载 {train_file}，共 {len(train_data)} 条记录")

        with open(val_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        print(f"已加载 {val_file}，共 {len(val_data)} 条记录")

        return cot_df, id_mapping, train_data, val_data

    except Exception as e:
        print(f"加载数据文件时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None, None, None


def update_json_data(train_data, val_data, cot_df, id_mapping):
    """更新JSON数据中的knowledge_code字段"""
    print("\n正在更新JSON数据...")

    try:
        # 1. 构建 challenge_id -> implicit_knowledge_ids 映射
        challenge_to_knowledge = {}
        
        for _, row in cot_df.iterrows():
            challenge_id = str(row.get('challenge_id', '')).strip()
            if not challenge_id or challenge_id.lower() == 'nan':
                continue
            
            knowledge_ids = parse_implicit_knowledge_ids(row.get('implicit_knowledge_ids', []))
            challenge_to_knowledge[challenge_id] = knowledge_ids

        print(f"已创建 {len(challenge_to_knowledge)} 个 challenge_id -> implicit_knowledge_ids 映射")
        
        non_empty = sum(1 for v in challenge_to_knowledge.values() if v)
        print(f"其中 {non_empty} 个有非空知识点")

        # 2. 构建 exer_id(new_id) -> challenge_id 映射
        # 映射格式: {challenge_id: {original_id, new_id}}
        exer_to_challenge = {}
        for challenge_id, info in id_mapping.items():
            try:
                new_id = int(info['new_id'])
                exer_to_challenge[new_id] = challenge_id
            except Exception as e:
                print(f"  警告: 解析映射失败: {challenge_id}: {info}, 错误: {e}")

        print(f"已创建 {len(exer_to_challenge)} 个 exer_id -> challenge_id 映射")

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

                if exer_id in exer_to_challenge:
                    challenge_id = exer_to_challenge[exer_id]

                    if challenge_id in challenge_to_knowledge:
                        knowledge_ids = challenge_to_knowledge[challenge_id]
                        item['knowledge_code'] = knowledge_ids

                        if knowledge_ids:
                            updated_count += 1
                        else:
                            empty_count += 1
                    else:
                        not_found_in_excel += 1
                        if not_found_in_excel <= 3:
                            print(f"[{split_name}] Excel中未找到: exer_id={exer_id}, challenge_id={challenge_id}")
                else:
                    not_found_in_mapping += 1
                    if not_found_in_mapping <= 3:
                        print(f"[{split_name}] mapping中未找到: exer_id={exer_id}")

            print(f"\n{split_name} 更新统计:")
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


def save_json_data(updated_train, updated_val, output_dir, suffix="_cot"):
    """保存更新后的JSON数据"""
    print("\n正在保存更新后的JSON数据...")

    try:
        train_output = os.path.join(output_dir, f"train_d{suffix}.json")
        with open(train_output, 'w', encoding='utf-8') as f:
            json.dump(updated_train, f, ensure_ascii=False, indent=4)
        print(f"已保存: {train_output}")

        val_output = os.path.join(output_dir, f"val_d{suffix}.json")
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
    
    parser = argparse.ArgumentParser(description='使用CoT隐式知识点更新Mooper数据集')
    parser.add_argument('--cot', type=str, 
                        default="cot_implicit_knowledge_t0_6.xlsx",
                        help='CoT预测结果文件')
    parser.add_argument('--mapping', type=str,
                        default="data/exercise_id_mapping.json",
                        help='exercise_id映射文件')
    parser.add_argument('--train', type=str,
                        default="data/train_d.json",
                        help='原始train_d.json文件')
    parser.add_argument('--val', type=str,
                        default="data/val_d.json",
                        help='原始val_d.json文件')
    parser.add_argument('--output-dir', type=str,
                        default="data",
                        help='输出目录')
    parser.add_argument('--suffix', type=str,
                        default="_cot",
                        help='输出文件后缀')
    
    args = parser.parse_args()

    print("=" * 60)
    print("使用CoT隐式知识点更新Mooper数据集")
    print("=" * 60)

    cot_df, id_mapping, train_data, val_data = load_data_files(
        cot_result_file=args.cot,
        mapping_file=args.mapping,
        train_file=args.train,
        val_file=args.val
    )
    
    if cot_df is None:
        print("加载数据失败，程序终止")
        return

    updated_train, updated_val = update_json_data(
        train_data, val_data, cot_df, id_mapping
    )
    
    if updated_train is None:
        print("更新数据失败，程序终止")
        return

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
