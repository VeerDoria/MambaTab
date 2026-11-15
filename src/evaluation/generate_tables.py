"""Generate formatted summary tables corresponding to all saved figures in results/vanilla/figures.
"""

import json
import os
import pandas as pd

def main():
    raw_path = "results/vanilla/vanilla_results_raw.json"
    if not os.path.exists(raw_path):
        print(f"File {raw_path} does not exist.")
        return

    with open(raw_path) as f:
        data = json.load(f)

    models = [
        "mambatab_d",
        "mambatab_t",
        "xgboost",
        "xgboost_2022",
        "logistic_regression",
        "ft_transformer",
        "dcn",
    ]
    model_names = {
        "mambatab_d": "MambaTab-D",
        "mambatab_t": "MambaTab-T",
        "xgboost": "XGBoost",
        "xgboost_2022": "XGBoost (2022 Paper)",
        "logistic_regression": "Logistic Regression",
        "ft_transformer": "FT-Transformer",
        "dcn": "DCN",
    }
    # Filter to models actually present in data
    models = [m for m in models if m in data]

    datasets = ["adult", "income", "blastchar", "credit_g", "cylinder_bands"]
    dataset_names = {
        "adult": "Adult (AD)",
        "income": "Income-1995 (IC)",
        "blastchar": "Blastchar (BL)",
        "credit_g": "Credit-g (CG)",
        "cylinder_bands": "Cylinder-bands (CB)",
    }
    dataset_abbrev = {
        "adult": "AD",
        "income": "IC",
        "blastchar": "BL",
        "credit_g": "CG",
        "cylinder_bands": "CB",
    }

    paper_auroc = {
        "MambaTab-D": {"AD": 0.906, "IC": 0.906, "BL": 0.852, "CG": 0.771, "CB": 0.862},
        "MambaTab-T": {"AD": 0.914, "IC": 0.920, "BL": 0.854, "CG": 0.801, "CB": 0.896},
        "XGBoost": {"AD": 0.912, "IC": 0.925, "BL": 0.821, "CG": 0.726, "CB": 0.892},
        "Logistic Regression": {"AD": 0.851, "IC": 0.860, "BL": 0.801, "CG": 0.720, "CB": 0.748},
        "FT-Transformer": {"AD": 0.913, "IC": 0.915, "BL": 0.841, "CG": 0.739, "CB": 0.862},
        "DCN": {"AD": 0.913, "IC": 0.915, "BL": 0.840, "CG": 0.739, "CB": 0.848},
    }

    radar_metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Bal Acc"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-Score"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("specificity", "Specificity"),
        ("log_loss", "Log Loss"),
        ("training_time", "Train Time (s)"),
    ]

    out_dir = "results/vanilla/tables"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Export CSVs
    for ds in datasets:
        abbrev = dataset_abbrev[ds]
        rows = []
        for m in models:
            row = {"Model": model_names[m]}
            for met_key, met_disp in radar_metrics:
                val = data[m][ds].get(met_key)
                if val is not None:
                    row[met_disp] = f"{val['mean']:.4f} ± {val['std']:.4f}"
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, f"radar_{abbrev}_table.csv"), index=False)

    for met in ["auroc", "auprc", "accuracy", "f1_score", "training_time"]:
        rows = []
        for m in models:
            row = {"Model": model_names[m]}
            for ds in datasets:
                abbrev = dataset_abbrev[ds]
                val = data[m][ds].get(met)
                if val is not None:
                    if met == "training_time":
                        row[abbrev] = f"{val['mean']:.2f}s ± {val['std']:.2f}s"
                    else:
                        row[abbrev] = f"{val['mean']:.4f} ± {val['std']:.4f}"
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, f"{met}_comparison_formatted.csv"), index=False)

    # 2. Build Markdown Document
    doc = []
    doc.append("# Vanilla Supervised Learning — Metrics Summary Tables\n")
    doc.append("This document contains exact numerical tables for all figures generated in `results/vanilla/figures/`.\n")

    # Section 1: AUROC
    doc.append("## 1. Global AUROC Comparison (Figures: `auroc_comparison.png` & `auroc_heatmap.png`)\n")
    doc.append("| Model | AD (Adult) | IC (Income-1995) | BL (Blastchar) | CG (Credit-g) | CB (Cylinder-bands) | Mean AUROC |")
    doc.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in models:
        name = model_names[m]
        cells = []
        means = []
        for ds in datasets:
            val = data[m][ds]["auroc"]
            cells.append(f"{val['mean']:.4f} ± {val['std']:.4f}")
            means.append(val["mean"])
        avg = sum(means) / len(means)
        doc.append(f"| **{name}** | {' | '.join(cells)} | **{avg:.4f}** |")
    doc.append("\n")

    # Side-by-side comparison with Paper Table II
    doc.append("### AUROC: Replicated vs. Paper Table II\n")
    doc.append("| Model | Setting | AD | IC | BL | CG | CB |")
    doc.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in models:
        name = model_names[m]
        p_dict = paper_auroc.get(name, paper_auroc.get("XGBoost") if "XGBoost" in name else None)
        if p_dict is not None:
            p_vals = [f"{p_dict[dataset_abbrev[ds]]:.3f}" for ds in datasets]
            doc.append(f"| **{name}** | Paper | {' | '.join(p_vals)} |")
        else:
            doc.append(f"| **{name}** | Paper | {' | '.join(['N/A' for _ in datasets])} |")
        o_vals = [f"{data[m][ds]['auroc']['mean']:.3f} ± {data[m][ds]['auroc']['std']:.3f}" for ds in datasets]
        doc.append(f"| | Replicated | {' | '.join(o_vals)} |")
    doc.append("\n---\n")

    # Section 2: AUPRC
    doc.append("## 2. Global AUPRC Comparison (Figure: `auprc_heatmap.png`)\n")
    doc.append("| Model | AD (Adult) | IC (Income-1995) | BL (Blastchar) | CG (Credit-g) | CB (Cylinder-bands) | Mean AUPRC |")
    doc.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in models:
        name = model_names[m]
        cells = []
        means = []
        for ds in datasets:
            val = data[m][ds]["auprc"]
            cells.append(f"{val['mean']:.4f} ± {val['std']:.4f}")
            means.append(val["mean"])
        avg = sum(means) / len(means)
        doc.append(f"| **{name}** | {' | '.join(cells)} | **{avg:.4f}** |")
    doc.append("\n---\n")

    # Section 3: Radar Charts
    doc.append("## 3. Dataset-Specific Multi-Metric Tables (Radar Charts)\n")
    for ds in datasets:
        d_name = dataset_names[ds]
        abbrev = dataset_abbrev[ds]
        doc.append(f"### Figure: `radar_{abbrev}.png` — {d_name}\n")
        header = "| Model | " + " | ".join([disp for _, disp in radar_metrics]) + " |"
        sep = "| :--- | " + " | ".join([":---:" for _ in radar_metrics]) + " |"
        doc.append(header)
        doc.append(sep)
        for m in models:
            name = model_names[m]
            vals = []
            for key, _ in radar_metrics:
                v = data[m][ds].get(key)
                if v is not None:
                    vals.append(f"{v['mean']:.4f} ± {v['std']:.4f}")
                else:
                    vals.append("-")
            doc.append(f"| **{name}** | {' | '.join(vals)} |")
        doc.append("\n")

    # Section 4: Training Time
    doc.append("## 4. Training Time Comparison (seconds)\n")
    doc.append("| Model | AD (Adult) | IC (Income-1995) | BL (Blastchar) | CG (Credit-g) | CB (Cylinder-bands) | Mean Time |")
    doc.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in models:
        name = model_names[m]
        cells = []
        means = []
        for ds in datasets:
            val = data[m][ds].get("training_time")
            if val is not None:
                cells.append(f"{val['mean']:.2f}s ± {val['std']:.2f}s")
                means.append(val["mean"])
            else:
                cells.append("-")
        if means:
            avg = sum(means) / len(means)
            doc.append(f"| **{name}** | {' | '.join(cells)} | **{avg:.2f}s** |")
        else:
            doc.append(f"| **{name}** | {' | '.join(cells)} | **-** |")
    doc.append("\n---\n")

    summary_md_path = os.path.join(out_dir, "METRICS_SUMMARY.md")
    with open(summary_md_path, "w") as f:
        f.write("\n".join(doc))

    print(f"Generated {summary_md_path} and CSV tables successfully.")

if __name__ == "__main__":
    main()
