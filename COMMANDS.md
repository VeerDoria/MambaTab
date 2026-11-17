# 📖 MambaTab Replication — Commands & Execution Guide

This document lists all commands to run, evaluate, and benchmark the MambaTab codebase across **Vanilla Supervised Learning** and **Feature-Incremental Learning (FIL)** on local machines and cloud platforms (Kaggle/Colab).

---

## 📁 Run Directory & Results Structure

Every time an experiment is launched, the system automatically creates a new isolated run directory (e.g., `run_1`, `run_2`, etc.) to **never overwrite previous results**:
* Vanilla runs are stored under: `results/vanilla/run_<N>/`
* FIL runs are stored under: `results/fil/run_<N>/`
* Regression runs are stored under: `results/regression/run_<N>/`
* A convenient `latest` pointer in the results folder always points to your most recent run.

Each run folder contains:
* `SUMMARY.md`: Clean, comprehensive markdown report with formatted tables for all metrics (AUROC, Accuracy, F1-Score, Balanced Accuracy, AUPRC, and Average Training Time in seconds).
* `comparison_*.csv`: Individual clean CSV tables for each metric.
* `raw_results.json`: Complete serialised metrics and seed details.

---

## ⚡ 1. Quick Verification / Smoke Tests (~30s)

Always run a quick 1-seed test first to verify environment setup before launching long runs:

```bash
# Vanilla smoke test (Credit-g, 1 seed, 2 models)
python3 experiments/run_vanilla.py --models mambatab_d xgboost_paper --datasets credit_g --seeds 1

# FIL smoke test (Credit-g, 1 seed, 2 models, 3 stages)
python3 experiments/run_fil.py --models mambatab_d xgboost_paper --datasets credit_g --seeds 1
```

---

## 🧪 2. Vanilla Supervised Learning (Table II Replication)

Evaluates static tabular classification performance across all datasets and seeds.

### A. Full Benchmark (All 7 Models, 5 Datasets, 5 Seeds = 175 runs)
```bash
python3 experiments/run_vanilla.py
```
*(Add `--download` if running for the first time without pre-cached datasets).*

### B. Compare MambaTab vs. XGBoost Variants
```bash
python3 experiments/run_vanilla.py --models mambatab_d mambatab_t xgboost xgboost_paper
```

### C. Fast Benchmark (3 Seeds instead of 5)
```bash
python3 experiments/run_vanilla.py --seeds 3
```

### D. Single Dataset Evaluation (e.g. Adult or Credit-g)
```bash
python3 experiments/run_vanilla.py --datasets adult
```

### E. Run in Background (Detached with log file)
```bash
nohup python3 experiments/run_vanilla.py > vanilla_run.log 2>&1 &

# Monitor live progress bar:
tail -f vanilla_run.log
```

---

## 📈 3. Feature-Incremental Learning (FIL) (Table III Replication)

Evaluates progressive feature addition across 3 stages ($s_1 \rightarrow s_1 \cup s_2 \rightarrow s_1 \cup s_2 \cup s_3$).

### A. Full FIL Benchmark (All Models, 5 Datasets, 5 Seeds, 3 Stages = 600 training cycles)
```bash
python3 experiments/run_fil.py
```
*Estimated runtime: ~1.5 hours on local GPU.*

### B. Compare Dynamic MambaTab vs. Paper Baseline Constraints
```bash
python3 experiments/run_fil.py --models mambatab_d mambatab_t xgboost xgboost_paper
```

### C. Fast FIL Run (1 or 3 Seeds)
```bash
python3 experiments/run_fil.py --seeds 3
```

### D. Run FIL in Background
```bash
nohup python3 experiments/run_fil.py > fil_run.log 2>&1 &

# Monitor live progress bar:
tail -f fil_run.log
```

---

## 📈 4. Tabular Regression Benchmark (Separate Experiment)

Evaluates continuous target prediction across **California Housing**, **Abalone**, and **Concrete Compressive Strength**.

### A. Full Regression Benchmark (All 7 Models, 3 Datasets, 5 Seeds = 105 runs)
```bash
python3 experiments/run_regression.py --download
```

### B. Quick Regression Smoke Test (Concrete, 1 seed, MambaTab-D vs. XGBoost Paper)
```bash
python3 experiments/run_regression.py --models mambatab_d xgboost_paper --datasets concrete --seeds 1
```

### C. Specific Models / Datasets
```bash
python3 experiments/run_regression.py --models mambatab_d mambatab_t xgboost xgboost_paper --datasets california_housing
```

### D. Run Regression in Background
```bash
nohup python3 experiments/run_regression.py > regression_run.log 2>&1 &
tail -f regression_run.log
```

---

## 📊 5. Post-Processing & Table Generation

Generate formatted markdown metric tables and side-by-side paper comparisons from completed runs:

```bash
# Generate METRICS_SUMMARY.md and formatted CSV tables for Vanilla
python3 src/evaluation/generate_tables.py
```
Outputs are saved to:
* `results/vanilla/tables/METRICS_SUMMARY.md`
* `results/vanilla/tables/*_comparison_formatted.csv`

---

## ☁️ 5. Cloud Execution (Kaggle Notebooks)

### Step 1: Notebook Setup
* **Accelerator**: GPU T4 x2 (or GPU P100)
* **Internet**: Toggle to **"Internet on"** (Settings panel on right sidebar)

### Step 2: In First Notebook Cell
```python
# Clone repo & enter folder
!git clone https://github.com/VeerDoria/MambaTab.git
%cd MambaTab

# Install dependencies (Kaggle has PyTorch/XGBoost preinstalled)
!pip install -r requirements.txt
```

### Step 3: Run Experiments in Next Cells
```python
# Run Vanilla benchmark:
!python3 experiments/run_vanilla.py --download

# Run FIL benchmark:
!python3 experiments/run_fil.py
```

### Step 4: Package & Download Results from Kaggle
```python
# Create downloadable zip in /kaggle/working:
!zip -r /kaggle/working/mambatab_results.zip results/
```
Download `mambatab_results.zip` directly from the **Output** section in the right sidebar.

---

## 📋 6. Reference Cheat Sheet

### Available Model Identifiers (`--models`)
| Identifier | Description | Setup Details |
| :--- | :--- | :--- |
| `mambatab_d` | **MambaTab-D** (Default) | $D=32, N=32, d_{\text{conv}}=4, M=1$, expansion=2 |
| `mambatab_t` | **MambaTab-T** (Tuned) | Dataset-specific optimal params from Paper Table VI |
| `xgboost` | **Modern XGBoost** (2026) | 300 estimators, early stopping, GPU accelerated |
| `xgboost_paper` | **XGBoost Paper Baseline** | 100 estimators, **no early stopping**, CPU, $s_1$-constrained in FIL |
| `logistic_regression` | **Logistic Regression** | $C=1.0$, L-BFGS solver, max 1000 iterations |
| `ft_transformer` | **FT-Transformer** | Feature Tokenizer + Transformer (NeurIPS 2021) |
| `dcn` | **Deep & Cross Network** | Cross polynomial interaction layers + Deep MLP |

### Available Datasets (`--datasets`)

**Classification Datasets:**
| Identifier | Abbreviation | Dataset Name | OpenML ID | Features | Samples |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `adult` | **AD** | Adult Census Income | 1590 | 14 | 48,842 |
| `income` | **IC** | Census-Income (KDD-95) | 4535 | 41 | 32,561 |
| `blastchar` | **BL** | Telco Customer Churn | 42178 | 19 | 7,043 |
| `credit_g` | **CG** | German Credit Risk | 31 | 20 | 1,000 |
| `cylinder_bands` | **CB** | Cylinder Bands | 6332 | 39 | 540 |

**Regression Datasets:**
| Identifier | Abbreviation | Dataset Name | Source / ID | Features | Samples | Target Variable |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| `california_housing` | **CH** | California Housing | Scikit / 531 | 8 | 20,640 | `MedHouseVal` |
| `abalone` | **AB** | Abalone | OpenML 183 | 8 | 4,177 | `Class_number_of_rings` |
| `concrete` | **CO** | Concrete Compressive Strength | OpenML 4353 | 8 | 1,030 | `compressive_strength` |

### CLI Arguments Summary
| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--models` | list | `all` | Specific models to evaluate |
| `--datasets` | list | `all` | Specific datasets to evaluate |
| `--seeds` | int | `5` | Number of random seeds (1 to 5) |
| `--stages` | int | `3` | Incremental feature stages (*FIL only*) |
| `--download` | flag | `False` | Auto-download and cache OpenML datasets |
| `--results-dir` | path | `results/{vanilla,fil}` | Directory where metrics and figures are stored |
| `--config` | path | `configs/default.yaml` | Path to custom YAML configuration file |
