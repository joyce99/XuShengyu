# Q-free_CD

This repository is the implementation and experiment codebase for the paper **Q-free CD: A Framework for Automatic Q-Matrix Generation and Structured Enhancement for Cognitive Diagnosis via LLMs**. It focuses on automatic Q-matrix generation, structured enhancement, and downstream cognitive diagnosis.

The core idea of the paper is to reduce the dependence on expert-annotated Q-matrices. Instead of treating the Q-matrix as a fixed prerequisite, Q-free CD uses LLMs to automatically generate and enhance it, moving cognitive diagnosis closer to a Q-free paradigm where response logs remain central and Q is constructed automatically.

From the paper's perspective, the framework follows a three-stage pipeline:

- `Qex`
  - Explicit Q-matrix generation
  - Captures shallow exercise-KC associations through semantic alignment
- `Qim`
  - Implicit Q-matrix generation
  - Recovers latent knowledge dependencies through CoT reasoning paths
- `Qen`
  - Enhanced Q-matrix
  - Integrates and completes Q through structural rules and knowledge composition

This repository is best understood as a paper-oriented research codebase rather than a packaged Python library. It already includes many intermediate artifacts and generated outputs, making it suitable for reproduction, follow-up experiments, and incremental processing.

## Overview

Following the paper, the goal is to automatically construct and enhance Q-matrices so that they encode not only explicit semantic associations, but also implicit reasoning signals and structured relations among knowledge concepts, and then use the generated Q-matrices for downstream cognitive diagnosis.

The paper emphasizes that the structural quality and reasoning quality of the Q-matrix are major bottlenecks for CD performance. Compared with traditional expert-built Q-matrices that are often sparse and flat, Q-free CD aims to produce a more complete, structured, and explainable cognitive representation.

Main scripts:

- `LLM.py` / `LLM_zp.py`
  - Predict explicit knowledge points from exercises.
  - Combine vector retrieval with LLM-based scoring.
- `build_knowledge_graph.py`
  - Build a knowledge graph from the full knowledge-point inventory.
  - Focus on hierarchy, composite relations, and domain grouping.
- `cot_knowledge_extractor.py`
  - Extract implicit knowledge points from CoT reasoning paths.
  - Map candidate premises back to the knowledge base through vector retrieval.
- `rule_based_qmatrix_enhancement.py`
  - Apply rule R1/R2 to merge explicit and implicit knowledge points.
  - Produce an enhanced Q-matrix result file.
- `update_*.py`
  - Write predicted or enhanced knowledge points back into the `knowledge_code` field of the train/validation data.

## Paper Correspondence

- Paper title
  - `Q-free CD: A Framework for Automatic Q-Matrix Generation and Structured Enhancement for Cognitive Diagnosis via LLMs`
- Target task
  - automatic Q-matrix generation for cognitive diagnosis
- Main methodological components
  - Explicit Q-Matrix Generation
  - Implicit Q-Matrix Generation
  - Rule-based Q-Matrix Enhancement
  - Cognitive Diagnosis with Generated Q-matrix

According to the paper abstract, the framework relies on LLM semantic understanding and reasoning ability to model shallow explicit associations, deep implicit associations, and structured rule-based completion, thereby producing a more comprehensive and explainable Q-matrix.

## Mapping the Paper to the Code

The following mapping helps connect the paper modules to the scripts in this repository:

- `Explicit Q-Matrix Generation` in the paper
  - implemented by `LLM.py` and `LLM_zp.py`
  - produces the explicit Q variant, `Qex`
- `Implicit Q-Matrix Generation` in the paper
  - implemented by `cot_knowledge_extractor.py`
  - produces the implicit Q variant, `Qim`
- `Rule-based Q-Matrix Enhancement` in the paper
  - implemented by `build_knowledge_graph.py` and `rule_based_qmatrix_enhancement.py`
  - the former builds structured relations, while the latter applies rules R1 / R2 to produce `Qen`
- `Cognitive Diagnosis Module` in the paper
  - connected through `update_knowledge_*.py`, `update_cot_knowledge.py`, and `update_enhanced_knowledge.py`
  - these scripts write the generated Q results back into dataset files for downstream CD models

## Datasets and Paper Experiments

The paper evaluates Q-free CD on three real-world datasets with exercise text:

- `MOOPer`
- `MOOCRadar`
- `NIPS20`

This repository currently contains well-organized code and data layout mainly for the first two:

- `Mooper/`
  - corresponds to `MOOPer` in the paper
- `MOOCRadar-middle/`
  - corresponds to `MOOCRadar` in the paper

In other words, this repository is best viewed as the practical implementation of the paper framework on `MOOPer` and `MOOCRadar`. The paper also reports results on `NIPS20`, but there is no dedicated sibling subproject for it in the current directory structure.

## Repository Layout

The root directory contains two mostly independent subprojects:

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

## Recommended Workflow

Both subprojects follow roughly the same workflow, which directly matches the paper pipeline:

1. Generate explicit Q (`Qex`)  
   Run `LLM.py` or `LLM_zp.py` to predict explicit knowledge associations from semantic alignment between exercises and KCs.

2. Build structural knowledge  
   Run `build_knowledge_graph.py` to build composite relations, hierarchy, and domain structure for later enhancement.

3. Generate implicit Q (`Qim`)  
   Run `cot_knowledge_extractor.py` to extract implicit knowledge from reasoning paths.

4. Generate enhanced Q (`Qen`)  
   Run `rule_based_qmatrix_enhancement.py` to merge explicit and implicit knowledge points and supplement composite concepts.

5. Connect to cognitive diagnosis data  
   Run `update_knowledge_*.py`, `update_cot_knowledge.py`, or `update_enhanced_knowledge.py` to write the generated Q results back into the dataset for downstream CD models.

If you only want to continue existing experiments, you usually do not need to rerun the full pipeline from scratch because many intermediate outputs are already stored in the repository.

## Requirements

- Python 3.9+
- Recommended packages:
  - `pandas`
  - `numpy`
  - `tqdm`
  - `openai`
  - `zhipuai`
  - `openpyxl`
  - `faiss-cpu` or another FAISS build compatible with your platform

A minimal setup example:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy tqdm openai zhipuai openpyxl faiss-cpu
```

## Model and API Configuration

Both subprojects contain a `config.py` file that is imported by the scripts.

The current code expects some combination of the following backends:

- A local OpenAI-compatible service at `http://127.0.0.1:12345/v1`
- DashScope-compatible endpoints for Qwen-family models
- ZhipuAI endpoints for GLM-family models
- A local or compatible embedding service for `text-embedding-bge-m3`

Before running the scripts, it is recommended to:

- Replace the keys and endpoints in `config.py` with your own settings
- Prefer environment variables over hard-coded secrets
- Make sure the local embedding service or local OpenAI-compatible service is already running

## Mooper Examples

The following commands assume you are working inside the `Mooper` directory.

### 1. Explicit knowledge prediction

```powershell
cd Mooper
python LLM_zp.py --input data/xlsx/filtered_results_with_summary.xlsx --threshold 0.6 --top_k 3 --analysis_k 30
```

The script usually generates output files such as:

- `prediction_results_unlimited_threshold0_6_zp.xlsx`
- or another variant with the same naming pattern

### 2. Build the knowledge graph

```powershell
python build_knowledge_graph.py --topics data/xlsx/topics.csv --output data/knowledge_graph.json --batch-size 20
```

### 3. Extract implicit knowledge

```powershell
python cot_knowledge_extractor.py --file data/xlsx/filtered_results_with_summary.xlsx --mapping data/exercise_id_mapping.json --threshold 0.6 --model qwen-max
```

### 4. Enhance the Q-matrix

```powershell
python rule_based_qmatrix_enhancement.py --explicit-file prediction_results_unlimited_threshold0_6_zp.xlsx --implicit-file cot_implicit_knowledge_t0_6.xlsx --output-file enhanced_qmatrix_results.xlsx --topics data/xlsx/topics.csv --knowledge-graph data/knowledge_graph.json
```

### 5. Write enhanced results back to the dataset

```powershell
python update_enhanced_knowledge.py --enhanced-file enhanced_qmatrix_results.xlsx --mapping-file data/exercise_id_mapping.json --train-file data/train_d.json --val-file data/val_d.json --output-dir data --knowledge-column enhanced_knowledge_ids --output-suffix enhanced
```

If you only want to write back explicit or implicit results, you can also use:

- `update_knowledge_mooper.py`
- `update_cot_knowledge.py`

## MOOCRadar-middle Examples

The following commands assume you are working inside the `MOOCRadar-middle` directory.

### 1. Explicit knowledge prediction

```powershell
cd MOOCRadar-middle
python LLM_zp.py --file data/problem_formatted.json --threshold 0.6 --unlimited
```

The script usually generates output files such as:

- `prediction_results_unlimited_threshold0_6_zp.xlsx`
- or another variant with the same naming pattern

### 2. Build the knowledge graph

```powershell
python build_knowledge_graph.py --concept-mapping data/concept_mapping.json --output data/knowledge_graph.json --batch-size 20 --mode domain
```

### 3. Extract implicit knowledge

```powershell
python cot_knowledge_extractor.py --file data/problem_formatted.json --mapping data/problem_id_mapping.json --threshold 0.6 --model qwen-max
```

### 4. Enhance the Q-matrix

```powershell
python rule_based_qmatrix_enhancement.py --explicit-file xlsx/prediction_results_unlimited_threshold0_6_zp.xlsx --implicit-file xlsx/cot_implicit_knowledge_t0_6.xlsx --output-file xlsx/enhanced_qmatrix_results.xlsx --concept-mapping data/concept_mapping.json --knowledge-graph data/knowledge_graph.json
```

### 5. Write enhanced results back to the dataset

```powershell
python update_enhanced_knowledge.py --enhanced-file xlsx/enhanced_qmatrix_results.xlsx --mapping-file data/problem_id_mapping.json --train-file data/MOOCRadar-middle/train.json --val-file data/MOOCRadar-middle/val.json --output-dir data/MOOCRadar-middle --knowledge-column enhanced_knowledge_ids --output-suffix enhanced
```

To write back only the implicit knowledge results:

```powershell
python update_cot_knowledge.py --cot xlsx/cot_implicit_knowledge_t0_6.xlsx --mapping data/problem_id_mapping.json --train data/MOOCRadar-middle/train.json --val data/MOOCRadar-middle/val.json --output-dir data/MOOCRadar-middle --suffix _cot
```

## Main Output Files

Typical outputs include:

- Explicit Q results: `prediction_results_*.xlsx`
- Implicit Q results: `cot_implicit_knowledge_t*.xlsx`
- Knowledge graph: `knowledge_graph.json`
- Enhanced Q-matrix results: `enhanced_qmatrix_results.xlsx`
- Updated train/validation files:
  - `train_d_*.json` / `val_d_*.json`
  - `train_*.json` / `val_*.json`

From the paper's point of view, these files correspond to:

- `Qex`
  - explicit semantic association outputs
- `Qim`
  - reasoning-derived implicit knowledge outputs
- `Qen`
  - the final structurally enhanced Q-matrix after applying R1 / R2

## Key Findings from the Paper

The most important takeaways reflected in this codebase are:

- Q-matrix quality directly affects CD performance
  - the paper identifies structural quality and reasoning quality of Q as key bottlenecks
- among the generated Q variants, `Qen` performs best
  - compared with explicit-only or implicit-only Q, the enhanced Q is more complete and stable
- the generated Q is compatible with multiple downstream CD models
  - the paper reports compatibility with models such as NCDM, RCD, and KCD

The paper further argues that future work should treat the Q-matrix as a reasoning-driven and evolvable cognitive representation, rather than a one-time static annotation.

## Notes

- The commands above assume you run them inside the corresponding subproject directory, not from the repository root.
- Many files in this repository are large. Regenerating them can take a long time and may incur API costs.
- `MOOCRadar-middle/update_knowledge_moocradar.py` still contains hard-coded Windows absolute paths. Edit those paths before using it, or prefer the parameterized scripts such as `update_enhanced_knowledge.py` and `update_cot_knowledge.py`.
- The FAISS indexes under `vector_index/` can be reused and do not need to be rebuilt every time.
- This repository does not currently provide a standardized package, test suite, or single unified entrypoint. It is best used as a script-based research workflow.

## Typical Use Cases

This repository is suitable for:

- reproducing the paper pipeline from `Qex` to `Qim` to `Qen`
- analyzing how explicit semantics, implicit reasoning, and structural rules jointly affect cognitive diagnosis
- extending the knowledge graph and composite rules on top of the current datasets
- preparing enhanced Q-matrices and datasets for cognitive diagnosis or educational data mining tasks
