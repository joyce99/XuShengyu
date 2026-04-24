# Q-free_CD

本仓库是论文 **Q-free CD: A Framework for Automatic Q-Matrix Generation and Structured Enhancement for Cognitive Diagnosis via LLMs** 的代码实现与实验脚本集合，重点对应论文中的自动 Q-matrix 生成、结构化增强，以及下游认知诊断流程。

论文的核心思想是：不再把专家标注的 Q-matrix 当作唯一前提，而是借助大语言模型自动生成并增强 Q-matrix，将认知诊断从传统的“响应日志 + 人工 Q-matrix”双输入范式，推进到更接近“响应日志为核心、Q 自动生成”的 Q-free 框架。

从论文方法上看，整个框架包含三阶段流水线：

- `Qex`
  - 显式 Q-matrix 生成
  - 通过语义对齐挖掘题目文本与知识点文本的显式关联
- `Qim`
  - 隐式 Q-matrix 生成
  - 通过 CoT 推理路径恢复题目背后的潜在知识点依赖
- `Qen`
  - 增强 Q-matrix
  - 通过规则融合与结构补全，得到更完整、更可解释的 Q-matrix

这个仓库更像一个论文复现与实验型代码仓库，而不是一个打包好的 Python 库。仓库中已经包含大量中间产物与导出结果，适合继续实验、复现实验链路，或在现有数据基础上做增量处理。

## 项目概览

按照论文定义，项目目标是自动构建并增强 Q-matrix，使其不仅覆盖显式语义关联，还能够编码隐式推理知识和知识点组合结构，最终将生成的 Q-matrix 用于下游认知诊断模型。

论文中强调，Q-matrix 的结构质量和推理质量是认知诊断性能的重要瓶颈；相比“扁平、稀疏、依赖专家经验”的传统 Q，Q-free CD 希望生成一个更完整、更结构化、更可演化的认知表示。

主要脚本职责如下：

- `LLM.py` / `LLM_zp.py`
  - 对习题进行显式知识点预测。
  - 结合向量检索与 LLM 评分，输出知识点候选结果。
- `build_knowledge_graph.py`
  - 基于知识点全集构建知识图谱。
  - 重点挖掘层级关系、组合关系和领域划分。
- `cot_knowledge_extractor.py`
  - 基于 CoT 推理路径抽取隐式知识点。
  - 使用向量索引把候选 premise 映射回知识点库。
- `rule_based_qmatrix_enhancement.py`
  - 使用规则 R1/R2 融合显式与隐式知识点。
  - 生成增强后的 Q-matrix 结果。
- `update_*.py`
  - 将预测或增强后的知识点写回 `train` / `val` 数据文件中的 `knowledge_code` 字段。

## 对应论文

- 论文标题
  - `Q-free CD: A Framework for Automatic Q-Matrix Generation and Structured Enhancement for Cognitive Diagnosis via LLMs`
- 论文任务
  - 自动生成 Q-matrix，并将其用于认知诊断
- 论文方法关键词
  - Explicit Q-Matrix Generation
  - Implicit Q-Matrix Generation
  - Rule-based Q-Matrix Enhancement
  - Cognitive Diagnosis with Generated Q-matrix

根据论文摘要，这一框架利用 LLM 的语义理解和推理能力，分别建模浅层显式关联、深层隐式关联和结构化规则补全，从而生成更全面、可解释的 Q-matrix。

## 论文与代码的对应关系

下表可以帮助你把论文模块和代码脚本对齐：

- 论文中的 `Explicit Q-Matrix Generation`
  - 对应 `LLM.py`、`LLM_zp.py`
  - 作用是生成显式知识点关联，形成 `Qex`
- 论文中的 `Implicit Q-Matrix Generation`
  - 对应 `cot_knowledge_extractor.py`
  - 作用是从推理路径中抽取隐式知识点，形成 `Qim`
- 论文中的 `Rule-based Q-Matrix Enhancement`
  - 对应 `build_knowledge_graph.py` 和 `rule_based_qmatrix_enhancement.py`
  - 前者负责构建知识图谱与组合关系，后者负责按规则 R1 / R2 进行融合增强，形成 `Qen`
- 论文中的 `Cognitive Diagnosis Module`
  - 对应 `update_knowledge_mooper.py`、`update_cot_knowledge.py`、`update_enhanced_knowledge.py` 等回写脚本
  - 作用是把生成的 Q 结果对接到训练/验证数据，供下游 CD 模型使用

## 数据集与论文实验

论文实验使用了三个带题目文本的信息化教育数据集：

- `MOOPer`
- `MOOCRadar`
- `NIPS20`

当前仓库中直接包含并组织好的主要是前两个数据集对应代码：

- `Mooper/`
  - 对应论文中的 `MOOPer`
- `MOOCRadar-middle/`
  - 对应论文中的 `MOOCRadar`

也就是说，这个仓库当前更像是论文完整框架在 `MOOPer` 和 `MOOCRadar` 上的落地版本；论文中提到的 `NIPS20` 数据与实验逻辑可以参照相同流程扩展，但当前目录中未单独提供一个并列的 `NIPS20` 子项目。

## 仓库结构

根目录下包含两个相对独立的子项目：

```text
Q-free_CD-main/
+-- Mooper/
|   +-- data/
|   |   +-- xlsx/
|   |   +-- vector_index/
|   |   +-- train_d*.json / val_d*.json
|   |   +-- exercise_id_mapping.json
|   |   `-- knowledge_graph.json
|   +-- LLM.py / LLM_zp.py
|   +-- build_knowledge_graph.py
|   +-- cot_knowledge_extractor.py
|   +-- rule_based_qmatrix_enhancement.py
|   `-- update_*.py
`-- MOOCRadar-middle/
    +-- data/
    |   +-- MOOCRadar-middle/
    |   +-- vector_index/
    |   +-- problem_formatted.json
    |   +-- problem_id_mapping.json
    |   +-- concept_mapping.json
    |   `-- knowledge_graph.json
    +-- LLM.py / LLM_zp.py
    +-- build_knowledge_graph.py
    +-- cot_knowledge_extractor.py
    +-- rule_based_qmatrix_enhancement.py
    `-- update_*.py
```

## 推荐流程

两个子项目的总体流程基本一致，也和论文中的三阶段管线一一对应：

1. 生成显式 Q (`Qex`)  
   使用 `LLM.py` 或 `LLM_zp.py`，从题目文本与知识点文本的语义对齐关系中预测显式知识点。

2. 构建结构知识  
   使用 `build_knowledge_graph.py`，为后续规则增强提供知识点组合关系、层次关系和领域划分。

3. 生成隐式 Q (`Qim`)  
   使用 `cot_knowledge_extractor.py`，基于推理路径抽取隐式知识点。

4. 生成增强 Q (`Qen`)  
   使用 `rule_based_qmatrix_enhancement.py` 融合显式与隐式知识点，并补充组合知识点。

5. 对接认知诊断数据  
   使用 `update_knowledge_*.py`、`update_cot_knowledge.py`、`update_enhanced_knowledge.py` 将结果写回数据集，供下游 CD 模型调用。

如果你只是想继续当前实验，通常不需要从头跑完整条链路，因为仓库里已经保存了很多中间产物和输出文件。

## 环境要求

- Python 3.9 及以上
- 推荐依赖：
  - `pandas`
  - `numpy`
  - `tqdm`
  - `openai`
  - `zhipuai`
  - `openpyxl`
  - `faiss-cpu` 或与你平台兼容的 FAISS 版本

一个最小安装示例如下：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy tqdm openai zhipuai openpyxl faiss-cpu
```

## 模型与接口配置

两个子项目都带有 `config.py`，脚本运行时会读取其中的模型接口配置。

当前代码默认依赖以下几类后端：

- 本地 OpenAI 兼容服务：`http://127.0.0.1:12345/v1`
- DashScope 兼容接口，用于 Qwen 类模型
- ZhipuAI 接口，用于 GLM 类模型
- 本地或兼容接口的 `text-embedding-bge-m3` 向量服务

建议在运行前检查以下内容：

- 将 `config.py` 中的密钥与地址替换为你自己的配置
- 更推荐使用环境变量，而不是把真实密钥直接写在仓库里
- 确认本地向量模型或本地 OpenAI 兼容服务已经启动

## Mooper 使用示例

以下命令默认在 `Mooper` 目录下执行。

### 1. 显式知识点预测

```powershell
cd Mooper
python LLM_zp.py --input data/xlsx/filtered_results_with_summary.xlsx --threshold 0.6 --top_k 3 --analysis_k 30
```

脚本通常会自动生成类似下面的结果文件：

- `prediction_results_unlimited_threshold0_6_zp.xlsx`
- 或其他同名模式变体

### 2. 构建知识图谱

```powershell
python build_knowledge_graph.py --topics data/xlsx/topics.csv --output data/knowledge_graph.json --batch-size 20
```

### 3. 抽取隐式知识点

```powershell
python cot_knowledge_extractor.py --file data/xlsx/filtered_results_with_summary.xlsx --mapping data/exercise_id_mapping.json --threshold 0.6 --model qwen-max
```

### 4. 规则增强 Q-matrix

```powershell
python rule_based_qmatrix_enhancement.py --explicit-file prediction_results_unlimited_threshold0_6_zp.xlsx --implicit-file cot_implicit_knowledge_t0_6.xlsx --output-file enhanced_qmatrix_results.xlsx --topics data/xlsx/topics.csv --knowledge-graph data/knowledge_graph.json
```

### 5. 回写增强结果

```powershell
python update_enhanced_knowledge.py --enhanced-file enhanced_qmatrix_results.xlsx --mapping-file data/exercise_id_mapping.json --train-file data/train_d.json --val-file data/val_d.json --output-dir data --knowledge-column enhanced_knowledge_ids --output-suffix enhanced
```

如需只回写显式或隐式结果，也可以使用：

- `update_knowledge_mooper.py`
- `update_cot_knowledge.py`

## MOOCRadar-middle 使用示例

以下命令默认在 `MOOCRadar-middle` 目录下执行。

### 1. 显式知识点预测

```powershell
cd MOOCRadar-middle
python LLM_zp.py --file data/problem_formatted.json --threshold 0.6 --unlimited
```

脚本通常会自动生成类似下面的结果文件：

- `prediction_results_unlimited_threshold0_6_zp.xlsx`
- 或其他同名模式变体

### 2. 构建知识图谱

```powershell
python build_knowledge_graph.py --concept-mapping data/concept_mapping.json --output data/knowledge_graph.json --batch-size 20 --mode domain
```

### 3. 抽取隐式知识点

```powershell
python cot_knowledge_extractor.py --file data/problem_formatted.json --mapping data/problem_id_mapping.json --threshold 0.6 --model qwen-max
```

### 4. 规则增强 Q-matrix

```powershell
python rule_based_qmatrix_enhancement.py --explicit-file xlsx/prediction_results_unlimited_threshold0_6_zp.xlsx --implicit-file xlsx/cot_implicit_knowledge_t0_6.xlsx --output-file xlsx/enhanced_qmatrix_results.xlsx --concept-mapping data/concept_mapping.json --knowledge-graph data/knowledge_graph.json
```

### 5. 回写增强结果

```powershell
python update_enhanced_knowledge.py --enhanced-file xlsx/enhanced_qmatrix_results.xlsx --mapping-file data/problem_id_mapping.json --train-file data/MOOCRadar-middle/train.json --val-file data/MOOCRadar-middle/val.json --output-dir data/MOOCRadar-middle --knowledge-column enhanced_knowledge_ids --output-suffix enhanced
```

如需只回写隐式知识点结果，可以使用：

```powershell
python update_cot_knowledge.py --cot xlsx/cot_implicit_knowledge_t0_6.xlsx --mapping data/problem_id_mapping.json --train data/MOOCRadar-middle/train.json --val data/MOOCRadar-middle/val.json --output-dir data/MOOCRadar-middle --suffix _cot
```

## 主要输出文件

常见输出如下：

- 显式 Q 结果：`prediction_results_*.xlsx`
- 隐式 Q 结果：`cot_implicit_knowledge_t*.xlsx`
- 知识图谱：`knowledge_graph.json`
- 增强 Q-matrix 结果：`enhanced_qmatrix_results.xlsx`
- 回写后的训练/验证集：
  - `train_d_*.json` / `val_d_*.json`
  - `train_*.json` / `val_*.json`

从论文角度理解，这些文件分别对应：

- `Qex`
  - 显式语义关联生成结果
- `Qim`
  - 基于推理链抽取的隐式知识点结果
- `Qen`
  - 经 R1 / R2 结构增强后的最终 Q-matrix

## 论文结论摘要

根据论文实验结果，README 中最值得把握的结论有三点：

- 生成式 Q-matrix 的质量会直接影响 CD 性能
  - 论文指出，Q 的结构质量和推理质量是认知诊断的关键瓶颈
- 三种 Q 变体中，增强后的 `Qen` 效果最好
  - 相比只用显式或只用隐式知识点，结构增强后的 Q 更稳定、更完整
- 论文方法对多种 CD 模型具有兼容性
  - 生成的 Q 不依赖某个单一诊断模型，可以对接 NCDM、RCD、KCD 等下游方法

论文还指出，未来应把 Q-matrix 视为“推理驱动、可演化的认知表示”，而不是一次性固定的静态标注。

## 注意事项

- 命令默认假设你在各自子项目目录下执行，而不是根目录。
- 仓库中的很多文件体积较大，重新生成可能耗时较久，也会产生额外 API 成本。
- `MOOCRadar-middle/update_knowledge_moocradar.py` 中仍包含固定的 Windows 绝对路径，直接运行前需要先手动修改；更建议优先使用带命令行参数的 `update_enhanced_knowledge.py` 和 `update_cot_knowledge.py`。
- `vector_index/` 下的 FAISS 索引可以复用，不必每次重建。
- 本仓库目前没有标准化打包、测试或统一入口，更适合按脚本逐步运行。

## 适用场景

这个仓库适合以下用途：

- 复现论文中的 `Qex -> Qim -> Qen` 自动 Q 生成流程
- 分析显式语义、隐式推理与结构规则三者如何共同影响认知诊断
- 在现有数据集上继续扩展知识图谱和组合规则
- 为认知诊断或教育数据挖掘任务准备增强版 Q-matrix 与数据集
