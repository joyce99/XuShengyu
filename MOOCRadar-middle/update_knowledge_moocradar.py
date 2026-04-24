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
            # 处理JSON格式的字符串，如"[1, 2, 3]"
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


# ====== 2. 加载所有数据文件（多了一个 problem_id_mapping.json） ======
def load_data_files():
    """加载所有需要的数据文件"""
    print("正在加载数据文件...")

    try:
        # 1) 加载预测结果 Excel
        knowledge_df = pd.read_excel(
            'D:/code/Qfree-MOOCRadar/xlsx-parameter/prediction_results_unlimited_threshold0_5_zp_1.xlsx'
        )
        print(f"已加载prediction_results_unlimited_threshold0_5_zp_1.xlsx，共{len(knowledge_df)}行")

        # 2) 加载 problem_id_mapping.json  (exer_id -> problem_id)
        with open('D:/code/Qfree-MOOCRadar/problem_id_mapping.json', 'r', encoding='utf-8') as f:
            problem_id_mapping = json.load(f)
        print(f"已加载problem_id_mapping.json，共{len(problem_id_mapping)}条映射（exer_id -> problem_id）")

        # 3) 加载 train.json / val.json
        with open('D:/code/Qfree-MOOCRadar/MOOCRadar-middle-parameter/train.json', 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"已加载train.json，共{len(train_data)}条记录")

        with open('D:/code/Qfree-MOOCRadar/MOOCRadar-middle-parameter/val.json', 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        print(f"已加载val.json，共{len(val_data)}条记录")

        return knowledge_df, problem_id_mapping, train_data, val_data

    except Exception as e:
        print(f"加载数据文件时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None, None, None


# ====== 3. 更新 JSON 中的 knowledge_code ======
def update_json_data(train_data, val_data, knowledge_df, problem_id_mapping):
    """更新JSON数据中的knowledge_code字段"""
    print("\n正在更新JSON数据...")

    try:
        # ---- 3.1 先构建 problem_id -> knowledge_ids 的映射表 ----
        # Excel 列名：problem_id, knowledge_ids
        problem_to_knowledge = {}

        for _, row in knowledge_df.iterrows():
            if 'problem_id' not in row or 'knowledge_ids' not in row:
                continue

            problem_id = str(row['problem_id']).strip()
            if problem_id == '' or problem_id.lower() == 'nan':
                continue

            knowledge_ids = parse_knowledge_ids(row['knowledge_ids'])
            problem_to_knowledge[problem_id] = knowledge_ids

        print(f"已创建 {len(problem_to_knowledge)} 个 problem_id -> knowledge_ids 的映射")

        # ---- 3.2 把 problem_id_mapping 转成 exer_id(int) -> problem_id 的 dict ----
        exer_to_problem = {}
        for k, v in problem_id_mapping.items():
            try:
                exer_id = int(k)  # key 是 "0", "1", ...
                problem_id = str(v).strip()
                exer_to_problem[exer_id] = problem_id
            except Exception as e:
                print(f"  警告: 解析 problem_id_mapping 条目失败: {k}: {v}, 错误: {e}")

        print(f"已创建 {len(exer_to_problem)} 个 exer_id -> problem_id 的映射")

        # ---- 3.3 定义一个通用的更新函数，给 train / val 共用 ----
        def update_one_split(data, split_name):
            updated = []
            updated_count = 0
            empty_count = 0
            not_found_count = 0
            no_excel_count = 0

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
                            else:
                                empty_count += 1
                        else:
                            # 有 exer_id -> problem_id 映射，但 Excel 中没有该 problem_id
                            no_excel_count += 1
                            if no_excel_count <= 5:
                                print(f"[{split_name}] problem_id 在 Excel 中未找到: "
                                      f"exer_id={exer_id}, problem_id={problem_id}")
                    else:
                        # 在 problem_id_mapping.json 中没找到这个 exer_id
                        not_found_count += 1
                        if not_found_count <= 5:
                            print(f"[{split_name}] 在 problem_id_mapping.json 中未找到 exer_id: {exer_id}")

                updated.append(item)

            print(f"\n已更新{split_name}.json中的knowledge_code字段:")
            print(f"  更新的记录数: {updated_count}")
            print(f"  设置为空的记录数(Excel中有该problem但knowledge_ids为空): {empty_count}")
            print(f"  在problem_id_mapping.json中未找到exer_id的记录数: {not_found_count}")
            print(f"  在Excel中未找到problem_id的记录数: {no_excel_count}")

            return updated

        updated_train = update_one_split(train_data, "train")
        updated_val = update_one_split(val_data, "val")

        return updated_train, updated_val

    except Exception as e:
        print(f"更新JSON数据时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None


# ====== 4. 保存 JSON（你原来的函数可以直接复用） ======
def save_json_data(updated_train, updated_val):
    """保存更新后的JSON数据，与原始文件格式保持一致"""
    print("\n正在保存更新后的JSON数据...")

    try:
        with open('D:/code/Qfree-MOOCRadar/MOOCRadar-middle-parameter/train.json', 'r', encoding='utf-8') as f:
            original_content = f.read()

        indent = None
        indent_char = " "
        indent_size = 4

        lines = original_content.split('\n')
        for i in range(1, min(20, len(lines))):
            line = lines[i]
            if line.strip().startswith('"') or line.strip().startswith('{') or line.strip().startswith('['):
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces > 0:
                    indent_size = leading_spaces
                    if '\t' in line[:leading_spaces]:
                        indent_char = '\t'
                        indent_size = line[:leading_spaces].count('\t')
                    break

        if indent_char == " ":
            indent = indent_size
        else:
            indent = '\t' * indent_size

        print(f"检测到原始文件使用 {indent_size} 个{indent_char}作为缩进")

        output_dir = "D:/code/Qfree-MOOCRadar/MOOCRadar-middle-parameter"

        train_output = os.path.join(output_dir, "train_updated_zp_1.json")
        with open(train_output, 'w', encoding='utf-8') as f:
            json.dump(updated_train, f, ensure_ascii=False, indent=indent)
        print(f"已保存更新后的文件: {train_output} (原始文件保持不变)")

        val_output = os.path.join(output_dir, "val_updated_zp_1.json")
        with open(val_output, 'w', encoding='utf-8') as f:
            json.dump(updated_val, f, ensure_ascii=False, indent=indent)
        print(f"已保存更新后的文件: {val_output} (原始文件保持不变)")

        return True

    except Exception as e:
        print(f"保存JSON数据时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return False


# ====== 5. 主函数 ======
def main():
    print("开始更新JSON文件中的knowledge_code字段...\n")

    knowledge_df, problem_id_mapping, train_data, val_data = load_data_files()
    if knowledge_df is None or problem_id_mapping is None or train_data is None or val_data is None:
        print("加载数据文件失败，程序终止")
        return

    updated_train, updated_val = update_json_data(train_data, val_data, knowledge_df, problem_id_mapping)
    if updated_train is None or updated_val is None:
        print("更新JSON数据失败，程序终止")
        return

    success = save_json_data(updated_train, updated_val)
    if not success:
        print("保存JSON数据失败，程序终止")
        return

    print("\n处理完成！")
    print("所有更新后的文件都已保存，原始文件保持不变。")


if __name__ == "__main__":
    main()
