#!/usr/bin/env python3
"""
Plot phase results from curated JSONL.

Input:
- curated_best.jsonl produced by the curation step

Outputs:
- tidy CSV
- dense baseline CSV
- retention CSV
- few-shot gain CSV
- separate PNG plots for each task / setting
- heatmaps

Usage:
    python plot_curated_results.py \
        --input artifacts/eval_jsonl/curated_best.jsonl \
        --outdir artifacts/plots_phase1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Curated JSONL input")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for plots and CSVs")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_task(task: str) -> str:
    t = str(task).strip().lower()
    if "synthetic_linear_icl" in t:
        return "synthetic_linear_icl"
    if "gsm8k" in t:
        return "gsm8k"
    if "mmlu" in t:
        return "mmlu"
    if "bbh" in t:
        return "bbh"
    return t


def infer_shot_setting(task: str, num_fewshot: Any) -> str:
    t = str(task).strip().lower()
    try:
        k = int(num_fewshot)
    except Exception:
        k = None

    if "zeroshot" in t or k == 0:
        return "zero-shot"
    return "few-shot"


def pretty_method(method: str) -> str:
    mapping = {
        "dense": "Dense",
        "wanda": "Wanda",
        "wandaplus": "Wanda++",
        "sparsegpt": "SparseGPT",
        "wanda_owl": "Wanda-OWL",
        "magnitude": "Magnitude",
    }
    return mapping.get(method, method)


def build_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()

    if df.empty:
        raise ValueError("Input curated JSONL is empty.")

    df["phase"] = df["phase"].astype(str)
    df["task_group"] = df["task"].apply(normalize_task)
    df["shot_setting"] = df.apply(lambda r: infer_shot_setting(r["task"], r["num_fewshot"]), axis=1)
    df["method_label"] = df["method"].apply(pretty_method)
    df["sparsity"] = pd.to_numeric(df["sparsity"], errors="coerce")
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df["num_fewshot"] = pd.to_numeric(df["num_fewshot"], errors="coerce")

    df["label"] = df.apply(
        lambda r: "Dense" if r["method"] == "dense" else f"{pretty_method(r['method'])} {int(r['sparsity'])}%",
        axis=1,
    )

    return df


def save_csvs(df: pd.DataFrame, outdir: Path) -> None:
    df.to_csv(outdir / "tidy_results.csv", index=False)


def make_dense_baselines(df: pd.DataFrame) -> pd.DataFrame:
    dense = df[df["method"] == "dense"].copy()
    dense = dense[["phase", "task_group", "shot_setting", "metric_value"]].rename(
        columns={"metric_value": "dense_metric_value"}
    )
    dense = dense.drop_duplicates(subset=["phase", "task_group", "shot_setting"])
    return dense


def make_retention(df: pd.DataFrame, dense_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(dense_df, on=["phase", "task_group", "shot_setting"], how="left")
    merged["retention"] = merged["metric_value"] / merged["dense_metric_value"]
    return merged


def make_fewshot_gain(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["phase", "method", "method_label", "sparsity", "task_group"]
    pivot = df.pivot_table(
        index=base_cols,
        columns="shot_setting",
        values="metric_value",
        aggfunc="first",
    ).reset_index()

    if "few-shot" not in pivot.columns:
        pivot["few-shot"] = pd.NA
    if "zero-shot" not in pivot.columns:
        pivot["zero-shot"] = pd.NA

    pivot["fewshot_gain"] = pivot["few-shot"] - pivot["zero-shot"]
    return pivot


def plot_lines(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    outpath: Path,
    ylabel: str,
    include_dense: bool = True,
) -> None:
    plt.figure(figsize=(8, 5))

    plot_df = df.copy()
    methods = sorted(plot_df["method"].unique(), key=lambda m: (m != "dense", m))

    for method in methods:
        sub = plot_df[plot_df["method"] == method].sort_values(by=x_col)
        if sub.empty:
            continue
        if not include_dense and method == "dense":
            continue
        plt.plot(sub[x_col], sub[y_col], marker="o", label=pretty_method(method))

    plt.xlabel("Sparsity")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_task_metric_curves(df: pd.DataFrame, outdir: Path) -> None:
    phase1 = df[df["phase"] == "phase1"].copy()
    tasks = ["mmlu", "bbh", "gsm8k"]

    for task in tasks:
        for shot in ["zero-shot", "few-shot"]:
            sub = phase1[(phase1["task_group"] == task) & (phase1["shot_setting"] == shot)].copy()
            if sub.empty:
                continue

            outpath = outdir / f"raw_{task}_{shot.replace('-', '')}.png"
            plot_lines(
                sub,
                x_col="sparsity",
                y_col="metric_value",
                title=f"{task.upper()} {shot}: performance vs sparsity",
                outpath=outpath,
                ylabel="Metric value",
                include_dense=True,
            )


def plot_retention_curves(ret_df: pd.DataFrame, outdir: Path) -> None:
    phase1 = ret_df[ret_df["phase"] == "phase1"].copy()
    tasks = ["mmlu", "bbh", "gsm8k"]

    for task in tasks:
        for shot in ["zero-shot", "few-shot"]:
            sub = phase1[(phase1["task_group"] == task) & (phase1["shot_setting"] == shot)].copy()
            if sub.empty:
                continue

            outpath = outdir / f"retention_{task}_{shot.replace('-', '')}.png"
            plot_lines(
                sub,
                x_col="sparsity",
                y_col="retention",
                title=f"{task.upper()} {shot}: retention vs sparsity",
                outpath=outpath,
                ylabel="Retention vs dense",
                include_dense=False,
            )


def plot_fewshot_gain(gain_df: pd.DataFrame, outdir: Path) -> None:
    phase1 = gain_df[gain_df["phase"] == "phase1"].copy()
    tasks = ["mmlu", "bbh", "gsm8k"]

    for task in tasks:
        sub = phase1[phase1["task_group"] == task].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(8, 5))
        methods = sorted(sub["method"].unique(), key=lambda m: (m != "dense", m))
        for method in methods:
            method_df = sub[sub["method"] == method].sort_values("sparsity")
            plt.plot(
                method_df["sparsity"],
                method_df["fewshot_gain"],
                marker="o",
                label=pretty_method(method),
            )

        plt.xlabel("Sparsity")
        plt.ylabel("Few-shot gain")
        plt.title(f"{task.upper()}: few-shot minus zero-shot")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / f"fewshot_gain_{task}.png", dpi=200)
        plt.close()


def plot_synthetic(df: pd.DataFrame, outdir: Path) -> None:
    phase1 = df[(df["phase"] == "phase1") & (df["task_group"] == "synthetic_linear_icl")].copy()
    if phase1.empty:
        return

    for k in sorted(phase1["num_fewshot"].dropna().unique()):
        sub = phase1[phase1["num_fewshot"] == k].copy()
        if sub.empty:
            continue

        outpath = outdir / f"synthetic_icl_{int(k)}shot.png"
        plot_lines(
            sub,
            x_col="sparsity",
            y_col="metric_value",
            title=f"Synthetic linear ICL ({int(k)}-shot): performance vs sparsity",
            outpath=outpath,
            ylabel="Accuracy",
            include_dense=True,
        )


def plot_heatmap(ret_df: pd.DataFrame, outdir: Path) -> None:
    phase1 = ret_df[
        (ret_df["phase"] == "phase1")
        & (ret_df["task_group"].isin(["mmlu", "bbh", "gsm8k"]))
        & (ret_df["shot_setting"] == "few-shot")
        & (ret_df["method"] != "dense")
    ].copy()

    if phase1.empty:
        return

    grouped = (
        phase1.groupby(["method_label", "sparsity"], as_index=False)["retention"]
        .mean()
    )

    table = grouped.pivot(index="method_label", columns="sparsity", values="retention")
    if table.empty:
        return

    plt.figure(figsize=(8, 4.5))
    plt.imshow(table.values, aspect="auto")
    plt.colorbar(label="Average retention")
    plt.xticks(range(len(table.columns)), table.columns)
    plt.yticks(range(len(table.index)), table.index)
    plt.title("Average few-shot retention across MMLU / BBH / GSM8K")
    plt.xlabel("Sparsity")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.savefig(outdir / "heatmap_avg_retention_fewshot.png", dpi=200)
    plt.close()


def save_summary_tables(df: pd.DataFrame, ret_df: pd.DataFrame, gain_df: pd.DataFrame, outdir: Path) -> None:
    dense = df[df["method"] == "dense"].copy()
    dense.to_csv(outdir / "dense_baselines.csv", index=False)

    ret_df.to_csv(outdir / "retention_results.csv", index=False)
    gain_df.to_csv(outdir / "fewshot_gain_results.csv", index=False)

    # Best method by task/sparsity/shot
    non_dense = df[df["method"] != "dense"].copy()
    if not non_dense.empty:
        idx = non_dense.groupby(["phase", "task_group", "shot_setting", "sparsity"])["metric_value"].idxmax()
        best = non_dense.loc[idx].sort_values(["phase", "task_group", "shot_setting", "sparsity"])
        best.to_csv(outdir / "best_method_by_task_sparsity.csv", index=False)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.input)
    df = build_dataframe(rows)

    save_csvs(df, args.outdir)

    dense_df = make_dense_baselines(df)
    ret_df = make_retention(df, dense_df)
    gain_df = make_fewshot_gain(df)

    plot_task_metric_curves(df, args.outdir)
    plot_retention_curves(ret_df, args.outdir)
    plot_fewshot_gain(gain_df, args.outdir)
    plot_synthetic(df, args.outdir)
    plot_heatmap(ret_df, args.outdir)

    save_summary_tables(df, ret_df, gain_df, args.outdir)

    print("=" * 80)
    print("PLOTTING COMPLETE")
    print("=" * 80)
    print(f"Input:   {args.input}")
    print(f"Output:  {args.outdir}")
    print("\nFiles created include:")
    print("- tidy_results.csv")
    print("- dense_baselines.csv")
    print("- retention_results.csv")
    print("- fewshot_gain_results.csv")
    print("- best_method_by_task_sparsity.csv")
    print("- raw_* plots")
    print("- retention_* plots")
    print("- fewshot_gain_* plots")
    print("- synthetic_icl_* plots")
    print("- heatmap_avg_retention_fewshot.png")


if __name__ == "__main__":
    main()