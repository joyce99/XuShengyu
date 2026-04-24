# -*- coding: utf-8 -*-
"""
将预测的知识点结果更新到训练/验证数据集的 knowledge_code 字段中
Mooper 数据集版本
"""

import pandas as pd
import json
import os
import re


def parse_knowledge_ids(knowledge_ids):
    """解析knowledge_ids字段，确保返回正确的数值数组格式"""
    if pd.isna(knowledge_ids) or knowledge_ids == '':
        return []

    try:
        if isinstance(knowledge_ids, str):
            if knowledge_ids.startswith('[') and knowledge_ids.endswith(']'):
                try:
                    knowledge_list = eval(knowledge_ids)
                    return [int(float(k)) for k in knowledge_list]
                except:
                    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', knowledge_ids)
                    return [int(float(n)) for n in numbers]
            else:
                parts = [p.strip() for p in knowledge_ids.split(',') if p.strip()]
                return [int(float(p)) for p in parts]

        elif isinstance(knowledge_ids, (list, tuple)):
            return [int(float(k)) for k in knowledge_ids]

        else:
            return [int(float(knowledge_ids))] if str(knowledge_ids).strip() != '' else []

    except Exception as e:
        print(f"  警告: 解析knowledge_ids时出错: {knowledge_ids}, 错误: {e}")
        return []


def load_data_files(excel_file, mapping_file, train_file, val_file):
    """加载所有需要的数据文件"""
    print("正在加载数据文件...")

    try:
        # 1) 加载预测结果 Excel
        knowledge_df = pd.read_excel(excel_file)
        print(f"已加载 {excel_file}，共 {len(knowledge_df)} 行")

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

        return knowledge_df, id_mapping, train_data, val_data

    except Exception as e:
        print(f"加载数据文件时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None, None, None


def update_json_data(train_data, val_data, knowledge_df, id_mapping):
    """更新JSON数据中的knowledge_code字段"""
    print("\n正在更新JSON数据...")

    try:
        # 1. 构建 challenge_id -> knowledge_ids 的映射表
        challenge_to_knowledge = {}
        for _, row in knowledge_df.iterrows():
            if 'challenge_id' not in row or 'knowledge_ids' not in row:
                continue

            challenge_id = str(row['challenge_id']).strip()
            if challenge_id == '' or challenge_id.lower() == 'nan':
                continue

            knowledge_ids = parse_knowledge_ids(row['knowledge_ids'])
            challenge_to_knowledge[challenge_id] = knowledge_ids

        print(f"已创建 {len(challenge_to_knowledge)} 个 challenge_id -> knowledge_ids 的映射")

        # 2. 构建 exer_id(new_id) -> challenge_id 的映射
        # 映射格式: {challenge_id: {original_id, new_id}}
        exer_to_challenge = {}
        for challenge_id, info in id_mapping.items():
            try:
                new_id = int(info['new_id'])  # new_id 就是 exer_id
                exer_to_challenge[new_id] = challenge_id
            except Exception as e:
                print(f"  警告: 解析映射条目失败: {challenge_id}: {info}, 错误: {e}")

        print(f"已创建 {len(exer_to_challenge)} 个 exer_id -> challenge_id 的映射")

        # 3. 定义更新函数
        def update_one_split(data, split_name):
            updated_count = 0
            empty_count = 0
            not_found_count = 0
            no_excel_count = 0

            for item in data:
                if 'exer_id' in item:
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
                            no_excel_count += 1
                            if no_excel_count <= 5:
                                print(f"[{split_name}] challenge_id 在 Excel 中未找到: "
                                      f"exer_id={exer_id}, challenge_id={challenge_id}")
                    else:
                        not_found_count += 1
                        if not_found_count <= 5:
                            print(f"[{split_name}] 在映射文件中未找到 exer_id: {exer_id}")

            print(f"\n已更新 {split_name} 中的 knowledge_code 字段:")
            print(f"  更新的记录数: {updated_count}")
            print(f"  设置为空的记录数: {empty_count}")
            print(f"  在映射文件中未找到 exer_id 的记录数: {not_found_count}")
            print(f"  在 Excel 中未找到 challenge_id 的记录数: {no_excel_count}")

            return data

        updated_train = update_one_split(train_data, "train")
        updated_val = update_one_split(val_data, "val")

        return updated_train, updated_val

    except Exception as e:
        print(f"更新JSON数据时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None


def save_json_data(updated_train, updated_val, output_dir, suffix="_updated"):
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

    parser = argparse.ArgumentParser(description='更新Mooper数据集的knowledge_code字段')
    parser.add_argument('--excel', type=str,
                        default='prediction_results_unlimited_threshold0_5.xlsx',
                        help='预测结果Excel文件路径')
    parser.add_argument('--mapping', type=str,
                        default='data/exercise_id_mapping.json',
                        help='习题ID映射文件路径')
    parser.add_argument('--train', type=str,
                        default='data/train_d.json',
                        help='训练数据JSON文件路径')
    parser.add_argument('--val', type=str,
                        default='data/val_d.json',
                        help='验证数据JSON文件路径')
    parser.add_argument('--output-dir', type=str,
                        default='data',
                        help='输出目录')
    parser.add_argument('--suffix', type=str,
                        default='_updated',
                        help='输出文件后缀')

    args = parser.parse_args()

    print("开始更新JSON文件中的knowledge_code字段...\n")

    knowledge_df, id_mapping, train_data, val_data = load_data_files(
        args.excel, args.mapping, args.train, args.val
    )
    if knowledge_df is None:
        print("加载数据文件失败，程序终止")
        return

    updated_train, updated_val = update_json_data(
        train_data, val_data, knowledge_df, id_mapping
    )
    if updated_train is None:
        print("更新JSON数据失败，程序终止")
        return

    success = save_json_data(updated_train, updated_val, args.output_dir, args.suffix)
    if not success:
        print("保存JSON数据失败，程序终止")
        return

    print("\n处理完成！原始文件保持不变。")


if __name__ == "__main__":
    main()
