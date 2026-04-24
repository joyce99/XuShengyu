"""
使用增强后的Q矩阵更新MOOCRadar数据集

输入:
- xlsx/enhanced_qmatrix_results.xlsx: 增强后的Q矩阵（R1+R2融合结果）
- problem_id_mapping.json: exer_id → problem_id 映射
- MOOCRadar-middle/train.json, val.json: 原始数据集

输出:
- MOOCRadar-middle/train_enhanced.json, val_enhanced.json: 更新后的数据集
"""

import pandas as pd
import json
import os
import re

# ====== 1. 解析 knowledge_ids 字段 ======
def parse_knowledge_ids(knowledge_ids):
    """解析knowledge_ids字段，确保返回正确的数值数组格式"""
    if pd.isna(knowledge_ids) or knowledge_ids == '':
        return []

    try:
        if isinstance(knowledge_ids, str):
            # 处理JSON格式的字符串，如"[1, 2, 3]" 或 '["1", "2", "3"]'
            if knowledge_ids.startswith('[') and knowledge_ids.endswith(']'):
                try:
                    knowledge_list = json.loads(knowledge_ids)
                    return [int(float(k)) for k in knowledge_list]
                except:
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


# ====== 2. 加载所有数据文件 ======
def load_data_files(enhanced_file: str, 
                    mapping_file: str,
                    train_file: str,
                    val_file: str):
    """加载所有需要的数据文件"""
    print("正在加载数据文件...")

    try:
        # 1) 加载增强后的Q矩阵 Excel
        knowledge_df = pd.read_excel(enhanced_file)
        print(f"已加载 {enhanced_file}，共{len(knowledge_df)}行")
        print(f"  列名: {list(knowledge_df.columns)}")

        # 2) 加载 problem_id_mapping.json (exer_id -> problem_id)
        with open(mapping_file, 'r', encoding='utf-8') as f:
            problem_id_mapping = json.load(f)
        print(f"已加载 {mapping_file}，共{len(problem_id_mapping)}条映射")

        # 3) 加载 train.json / val.json
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"已加载 {train_file}，共{len(train_data)}条记录")

        with open(val_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        print(f"已加载 {val_file}，共{len(val_data)}条记录")

        return knowledge_df, problem_id_mapping, train_data, val_data

    except Exception as e:
        print(f"加载数据文件时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None, None, None


# ====== 3. 更新 JSON 中的 knowledge_code ======
def update_json_data(train_data, val_data, knowledge_df, problem_id_mapping,
                     knowledge_column: str = 'enhanced_knowledge_ids'):
    """更新JSON数据中的knowledge_code字段"""
    print(f"\n正在更新JSON数据（使用 {knowledge_column} 列）...")

    try:
        # ---- 3.1 构建 problem_id -> knowledge_ids 的映射表 ----
        problem_to_knowledge = {}

        for _, row in knowledge_df.iterrows():
            if 'problem_id' not in row or knowledge_column not in row:
                continue

            problem_id = str(row['problem_id']).strip()
            if problem_id == '' or problem_id.lower() == 'nan':
                continue

            knowledge_ids = parse_knowledge_ids(row[knowledge_column])
            problem_to_knowledge[problem_id] = knowledge_ids

        print(f"已创建 {len(problem_to_knowledge)} 个 problem_id -> knowledge_ids 的映射")
        
        # 打印一些示例
        sample_items = list(problem_to_knowledge.items())[:3]
        for pid, kcs in sample_items:
            print(f"  示例: {pid} -> {len(kcs)} 个知识点")

        # ---- 3.2 把 problem_id_mapping 转成 exer_id(int) -> problem_id 的 dict ----
        exer_to_problem = {}
        for k, v in problem_id_mapping.items():
            try:
                exer_id = int(k)
                problem_id = str(v).strip()
                exer_to_problem[exer_id] = problem_id
            except Exception as e:
                print(f"  警告: 解析 problem_id_mapping 条目失败: {k}: {v}, 错误: {e}")

        print(f"已创建 {len(exer_to_problem)} 个 exer_id -> problem_id 的映射")

        # ---- 3.3 定义更新函数 ----
        def update_one_split(data, split_name):
            updated = []
            updated_count = 0
            empty_count = 0
            not_found_count = 0
            no_excel_count = 0
            total_kc_count = 0

            for item in data:
                if 'exer_id' in item:
                    exer_id = int(item['exer_id'])

                    if exer_id in exer_to_problem:
                        problem_id = exer_to_problem[exer_id]

                        if problem_id in problem_to_knowledge:
                            knowledge_ids = problem_to_knowledge[problem_id]
                            item['knowledge_code'] = knowledge_ids

                            if knowledge_ids:
                                updated_count += 1
                                total_kc_count += len(knowledge_ids)
                            else:
                                empty_count += 1
                        else:
                            no_excel_count += 1
                            if no_excel_count <= 3:
                                print(f"[{split_name}] problem_id 在 Excel 中未找到: "
                                      f"exer_id={exer_id}, problem_id={problem_id}")
                    else:
                        not_found_count += 1
                        if not_found_count <= 3:
                            print(f"[{split_name}] 在 problem_id_mapping.json 中未找到 exer_id: {exer_id}")

                updated.append(item)

            print(f"\n已更新 {split_name} 中的 knowledge_code 字段:")
            print(f"  成功更新记录数: {updated_count}")
            print(f"  平均每条记录知识点数: {total_kc_count / max(updated_count, 1):.2f}")
            print(f"  设置为空的记录数: {empty_count}")
            print(f"  未找到 exer_id 映射: {not_found_count}")
            print(f"  未找到 problem_id 映射: {no_excel_count}")

            return updated

        updated_train = update_one_split(train_data, "train")
        updated_val = update_one_split(val_data, "val")

        return updated_train, updated_val

    except Exception as e:
        print(f"更新JSON数据时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None


# ====== 4. 保存 JSON ======
def save_json_data(updated_train, updated_val, output_dir: str, suffix: str = "enhanced"):
    """保存更新后的JSON数据"""
    print("\n正在保存更新后的JSON数据...")

    try:
        os.makedirs(output_dir, exist_ok=True)
        
        train_output = os.path.join(output_dir, f"train_{suffix}.json")
        with open(train_output, 'w', encoding='utf-8') as f:
            json.dump(updated_train, f, ensure_ascii=False, indent=4)
        print(f"已保存: {train_output}")

        val_output = os.path.join(output_dir, f"val_{suffix}.json")
        with open(val_output, 'w', encoding='utf-8') as f:
            json.dump(updated_val, f, ensure_ascii=False, indent=4)
        print(f"已保存: {val_output}")

        return True

    except Exception as e:
        print(f"保存JSON数据时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return False


# ====== 5. 主函数 ======
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='使用增强Q矩阵更新MOOCRadar数据集')
    parser.add_argument('--enhanced-file', type=str, 
                        default='xlsx/enhanced_qmatrix_results.xlsx',
                        help='增强后的Q矩阵文件路径')
    parser.add_argument('--mapping-file', type=str,
                        default='problem_id_mapping.json',
                        help='ID映射文件路径')
    parser.add_argument('--train-file', type=str,
                        default='MOOCRadar-middle/train.json',
                        help='训练集文件路径')
    parser.add_argument('--val-file', type=str,
                        default='MOOCRadar-middle/val.json',
                        help='验证集文件路径')
    parser.add_argument('--output-dir', type=str,
                        default='MOOCRadar-middle',
                        help='输出目录')
    parser.add_argument('--knowledge-column', type=str,
                        default='enhanced_knowledge_ids',
                        help='要使用的知识点列名')
    parser.add_argument('--output-suffix', type=str,
                        default='enhanced',
                        help='输出文件后缀')
    
    args = parser.parse_args()
    
    print("="*60)
    print("使用增强Q矩阵更新MOOCRadar数据集")
    print("="*60)
    print(f"增强Q矩阵文件: {args.enhanced_file}")
    print(f"知识点列: {args.knowledge_column}")
    print(f"输出后缀: {args.output_suffix}")
    print("="*60 + "\n")

    # 加载数据
    knowledge_df, problem_id_mapping, train_data, val_data = load_data_files(
        args.enhanced_file,
        args.mapping_file,
        args.train_file,
        args.val_file
    )
    
    if knowledge_df is None:
        print("加载数据文件失败，程序终止")
        return

    # 更新数据
    updated_train, updated_val = update_json_data(
        train_data, val_data, 
        knowledge_df, problem_id_mapping,
        knowledge_column=args.knowledge_column
    )
    
    if updated_train is None:
        print("更新JSON数据失败，程序终止")
        return

    # 保存结果
    success = save_json_data(
        updated_train, updated_val, 
        args.output_dir,
        suffix=args.output_suffix
    )
    
    if not success:
        print("保存JSON数据失败，程序终止")
        return

    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)


if __name__ == "__main__":
    main()
