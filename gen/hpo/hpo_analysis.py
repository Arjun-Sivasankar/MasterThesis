#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HPO Console Log Analyzer — Professional (Matplotlib)

Usage:
  python hpo_analysis.py \
    --log_file /path/to/hpo_console.log \
    --out_dir  out/hpo_run_001

What you get in --out_dir:
  - trials.csv                  : tidy table (one row per trial)
  - summary.json                : best trial + counts
  - report.md                   : quick text recap
  - figs/*.png                  : professional static images using matplotlib
"""

import argparse
import ast
import json
import math
import re
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set professional matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.dpi'] = 120

# ---------------------------- parsing ----------------------------

def parse_log_text(text: str) -> Dict[str, Any]:
    """
    Parse HPO console log text into:
      - trials: list of trial dicts with params/metrics/timings/state
      - best:   best trial info (from the tail "Optimization Completed" section)
    """
    lines = text.splitlines()
    trials: List[Dict[str, Any]] = []
    curr: Optional[Dict[str, Any]] = None

    # Patterns learned from your logs
    re_start = re.compile(r"^.*=== Starting Trial\s+#(\d+)\s+===.*$")
    re_params = re.compile(r"^\[.*\]\s*Parameters:\s*(\{.*\})\s*$")
    re_train_done = re.compile(r"^\[.*\]\s*Training completed in\s*([\d\.]+)s")
    re_gen_done   = re.compile(r"^\[.*\]\s*Generation complete in\s*([\d\.]+)s")
    re_disk_usage = re.compile(r"^\[.*\]\s*Trial\s+(\d+)\s+disk usage:\s+([\d\.]+)\s*MB")
    re_micro      = re.compile(r"^\[.*\]\s*Trial\s+(\d+)\s+micro_f1:\s*([\d\.]+)")
    re_pruned     = re.compile(r"^\[.*\]\s*Trial\s+(\d+)\s+pruned\b", re.IGNORECASE)
    re_eval_loss_line = re.compile(r"\{.*'eval_loss':\s*([\d\.Ee+-]+).*?\}")

    # Tail summary
    re_best_trial  = re.compile(r"^\[.*\]\s*Best trial:\s*#(\d+)")
    re_best_metric = re.compile(r"^\[.*\]\s*Best\s+(\w+):\s*([\d\.]+)")
    re_best_params = re.compile(r"^\[.*\]\s*Best parameters:\s*(\{.*\})\s*$")

    best_trial_num: Optional[int] = None
    best_metric_value: Optional[float] = None
    best_params: Optional[Dict[str, Any]] = None

    last_eval_loss: Optional[float] = None

    for ln in lines:
        m = re_start.match(ln)
        if m:
            if curr is not None:
                curr.setdefault("state", "COMPLETED" if "micro_f1" in curr else "UNKNOWN")
                if curr.get("state") == "PRUNED" and "micro_f1" not in curr and last_eval_loss is not None:
                    curr["last_eval_loss_before_prune"] = last_eval_loss
                trials.append(curr)
            curr = {"trial": int(m.group(1)), "state": "RUNNING", "params": {}}
            last_eval_loss = None
            continue

        # If no current trial, we might be in the tail summary
        if curr is None:
            mbt = re_best_trial.match(ln)
            if mbt:
                try:
                    best_trial_num = int(mbt.group(1))
                except Exception:
                    pass
            mbm = re_best_metric.match(ln)
            if mbm:
                try:
                    best_metric_value = float(mbm.group(2))
                except Exception:
                    pass
            mbp = re_best_params.match(ln)
            if mbp:
                try:
                    best_params = ast.literal_eval(mbp.group(1))
                except Exception:
                    best_params = None
            continue

        mp = re_params.match(ln)
        if mp:
            try:
                params = ast.literal_eval(mp.group(1))
                if isinstance(params, dict):
                    curr["params"] = params
            except Exception:
                pass
            continue

        mt = re_train_done.match(ln)
        if mt:
            try:
                curr["train_seconds"] = float(mt.group(1))
            except Exception:
                pass
            continue

        mg = re_gen_done.match(ln)
        if mg:
            try:
                curr["gen_seconds"] = float(mg.group(1))
            except Exception:
                pass
            continue

        md = re_disk_usage.match(ln)
        if md:
            try:
                tn = int(md.group(1))
                if tn == curr.get("trial"):
                    curr["disk_mb"] = float(md.group(2))
            except Exception:
                pass
            continue

        mm = re_micro.match(ln)
        if mm:
            try:
                tn = int(mm.group(1))
                if tn == curr.get("trial"):
                    curr["micro_f1"] = float(mm.group(2))
                    curr["state"] = "COMPLETED"
            except Exception:
                pass
            continue

        mpn = re_pruned.match(ln)
        if mpn:
            try:
                tn = int(mpn.group(1))
                if tn == curr.get("trial"):
                    curr["state"] = "PRUNED"
            except Exception:
                pass
            continue

        mel = re_eval_loss_line.search(ln)
        if mel:
            try:
                last_eval_loss = float(mel.group(1))
                curr["last_eval_loss"] = last_eval_loss
            except Exception:
                pass
            continue

    if curr is not None:
        curr.setdefault("state", "COMPLETED" if "micro_f1" in curr else "UNKNOWN")
        if curr.get("state") == "PRUNED" and "micro_f1" not in curr and last_eval_loss is not None:
            curr["last_eval_loss_before_prune"] = last_eval_loss
        trials.append(curr)

    return {
        "trials": trials,
        "best": {"trial": best_trial_num, "metric_value": best_metric_value, "params": best_params},
    }


def parsed_to_df(parsed: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for t in parsed["trials"]:
        base = {
            "trial": t.get("trial"),
            "state": t.get("state"),
            "micro_f1": t.get("micro_f1", np.nan),
            "train_seconds": t.get("train_seconds", np.nan),
            "gen_seconds": t.get("gen_seconds", np.nan),
            "disk_mb": t.get("disk_mb", np.nan),
            "last_eval_loss": t.get("last_eval_loss", np.nan),
            "last_eval_loss_before_prune": t.get("last_eval_loss_before_prune", np.nan),
        }
        p = t.get("params", {})
        base.update({
            "epochs": p.get("epochs", np.nan),
            "learning_rate": p.get("learning_rate", np.nan),
            "warmup_ratio": p.get("warmup_ratio", np.nan),
            "weight_decay": p.get("weight_decay", np.nan),
            "grad_accum": p.get("grad_accum", np.nan),
            "lora_r": p.get("lora_r", np.nan),
            "lora_alpha": p.get("lora_alpha", np.nan),
            "lora_dropout": p.get("lora_dropout", np.nan),
            "patience": p.get("patience", np.nan),
            "per_device_train_batch_size": p.get("per_device_train_batch_size", np.nan),
            "per_device_eval_batch_size": p.get("per_device_eval_batch_size", np.nan),
        })
        rows.append(base)
    df = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    return df


# ---------------------------- figures ----------------------------

def make_matplotlib_figures(df: pd.DataFrame, figs_dir: Path) -> None:
    """
    Create and save professional matplotlib figures directly to the figs directory
    """
    comp = df[df["state"] == "COMPLETED"].copy()
    pruned = df[df["state"] == "PRUNED"].copy()
    
    if comp.empty:
        print("No completed trials found. Cannot generate figures.")
        return
        
    # Set global figure parameters
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # Color palette for consistency across plots
    palette = sns.color_palette("viridis", n_colors=8)
    
    # 1. micro_f1 by trial (bar chart)
    plt.figure(figsize=(10, 6))
    
    # Fix the deprecated barplot warning by using hue parameter
    comp['trial_str'] = comp['trial'].astype(str)
    ax = sns.barplot(x="trial_str", y="micro_f1", data=comp, hue="trial_str", legend=False)
    
    plt.title("Micro F1 Score by Trial", fontweight='bold')
    plt.xlabel("Trial Number")
    plt.ylabel("Micro F1 Score")
    
    # Add value annotations
    for i, row in enumerate(comp.iterrows()):
        _, r = row
        ax.text(
            i, r["micro_f1"] + 0.003, 
            f"{r['micro_f1']:.4f}", 
            ha='center', va='bottom', 
            fontsize=9, fontweight='semibold'
        )
    
    # Add a horizontal line for the average F1
    avg_f1 = comp['micro_f1'].mean()
    ax.axhline(y=avg_f1, color='red', linestyle='--', alpha=0.7)
    ax.text(
        len(comp) - 1, avg_f1 + 0.003, 
        f"Mean: {avg_f1:.4f}", 
        ha='right', va='bottom',
        color='darkred', fontsize=9, fontweight='semibold'
    )
    
    plt.tight_layout()
    plt.savefig(figs_dir / "micro_f1_by_trial.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figs_dir / 'micro_f1_by_trial.png'}")
    
    # 2. Learning rate vs micro_f1
    if comp["learning_rate"].notna().any():
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        scatter = plt.scatter(
            comp["learning_rate"], 
            comp["micro_f1"],
            c=comp["epochs"], 
            s=comp["micro_f1"] * 500,  # Size proportional to micro_f1
            cmap="viridis",
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add trial labels
        for i, row in comp.iterrows():
            plt.annotate(
                f"#{int(row['trial'])}", 
                (row["learning_rate"], row["micro_f1"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha='center', 
                fontsize=9,
                fontweight='semibold'
            )
        
        # Add colorbar with better formatting
        cbar = plt.colorbar(scatter)
        cbar.set_label('Number of Epochs', fontsize=11)
        
        plt.xscale("log")
        plt.title("Learning Rate Impact on Performance", fontweight='bold')
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Micro F1 Score")
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Mark the best trial
        best_row = comp.loc[comp['micro_f1'].idxmax()]
        plt.scatter(
            best_row['learning_rate'], 
            best_row['micro_f1'],
            s=200, 
            facecolors='none', 
            edgecolors='red', 
            linewidth=2,
            zorder=10
        )
        
        plt.tight_layout()
        plt.savefig(figs_dir / "lr_vs_f1.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {figs_dir / 'lr_vs_f1.png'}")
    
    # 3. Epochs vs micro_f1
    if comp["epochs"].notna().any():
        plt.figure(figsize=(10, 6))
        
        # Create discrete colors for grad_accum
        grad_accum_values = sorted(comp["grad_accum"].unique())
        
        # Fix the deprecated get_cmap warning
        colors = sns.color_palette("viridis", len(grad_accum_values))
        colors_dict = {val: colors[i] for i, val in enumerate(grad_accum_values)}
        
        # Create scatter plot with colors by grad_accum
        for ga in grad_accum_values:
            subset = comp[comp["grad_accum"] == ga]
            plt.scatter(
                subset["epochs"],
                subset["micro_f1"],
                s=subset["micro_f1"] * 500,  # Size proportional to micro_f1
                color=colors_dict[ga],
                alpha=0.8,
                edgecolor='white',
                linewidth=0.5,
                label=f"Grad Accum = {int(ga)}"
            )
        
        # Add trial labels
        for i, row in comp.iterrows():
            plt.annotate(
                f"#{int(row['trial'])}", 
                (row["epochs"], row["micro_f1"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha='center', 
                fontsize=9,
                fontweight='semibold'
            )
            
        plt.title("Impact of Training Duration on Performance", fontweight='bold')
        plt.xlabel("Number of Epochs")
        plt.ylabel("Micro F1 Score")
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend with better formatting
        legend = plt.legend(title="Gradient Accumulation", frameon=True, 
                 fontsize=9, title_fontsize=10)
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('lightgray')
        
        plt.tight_layout()
        plt.savefig(figs_dir / "epochs_vs_f1.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {figs_dir / 'epochs_vs_f1.png'}")
        
    # 4. LoRA hyperparams vs micro_f1 (two panels)
    if comp["lora_r"].notna().any() and comp["lora_dropout"].notna().any():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
        
        # Left subplot - LoRA r
        alpha_values = sorted(comp["lora_alpha"].unique())
        colors_alpha = sns.color_palette("viridis", len(alpha_values))
        colors_alpha_dict = {val: colors_alpha[i] for i, val in enumerate(alpha_values)}
        
        for alpha_val in alpha_values:
            alpha_df = comp[comp["lora_alpha"] == alpha_val]
            ax1.scatter(
                alpha_df["lora_r"],
                alpha_df["micro_f1"],
                s=120,
                color=colors_alpha_dict[alpha_val],
                label=f"α = {int(alpha_val)}",
                alpha=0.8,
                edgecolor='white',
                linewidth=0.5
            )
            
            # Add trial labels
            for i, row in alpha_df.iterrows():
                ax1.annotate(
                    f"#{int(row['trial'])}", 
                    (row["lora_r"], row["micro_f1"]),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha='center', 
                    fontsize=9,
                    fontweight='semibold'
                )
        
        ax1.set_title("LoRA Rank Parameter (r)", fontweight='bold')
        ax1.set_xlabel("LoRA Rank (r)")
        ax1.set_ylabel("Micro F1 Score")
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Better legend formatting
        legend1 = ax1.legend(title="LoRA Alpha", frameon=True, fontsize=9, title_fontsize=10)
        legend1.get_frame().set_alpha(0.9)
        legend1.get_frame().set_edgecolor('lightgray')
        
        # Right subplot - LoRA dropout
        r_values = sorted(comp["lora_r"].unique())
        colors_r = sns.color_palette("viridis", len(r_values))
        colors_r_dict = {val: colors_r[i] for i, val in enumerate(r_values)}
        
        for r_val in r_values:
            r_df = comp[comp["lora_r"] == r_val]
            ax2.scatter(
                r_df["lora_dropout"],
                r_df["micro_f1"],
                s=120,
                color=colors_r_dict[r_val],
                label=f"r = {int(r_val)}",
                alpha=0.8,
                edgecolor='white',
                linewidth=0.5
            )
            
            # Add trial labels
            for i, row in r_df.iterrows():
                ax2.annotate(
                    f"#{int(row['trial'])}", 
                    (row["lora_dropout"], row["micro_f1"]),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha='center', 
                    fontsize=9,
                    fontweight='semibold'
                )
        
        ax2.set_title("LoRA Dropout Parameter", fontweight='bold')
        ax2.set_xlabel("LoRA Dropout Rate")
        ax2.set_ylabel("Micro F1 Score")
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Better legend formatting
        legend2 = ax2.legend(title="LoRA Rank", frameon=True, fontsize=9, title_fontsize=10)
        legend2.get_frame().set_alpha(0.9)
        legend2.get_frame().set_edgecolor('lightgray')
        
        plt.suptitle("Impact of LoRA Parameters on Model Performance", fontweight='bold', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for the suptitle
        plt.savefig(figs_dir / "lora_params_vs_f1.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {figs_dir / 'lora_params_vs_f1.png'}")
    
    # 5. Weight decay vs micro_f1
    if comp["weight_decay"].notna().any():
        plt.figure(figsize=(10, 6))
        
        # Create normalized colormap for warmup_ratio
        norm = plt.Normalize(comp["warmup_ratio"].min(), comp["warmup_ratio"].max())
        
        # Fix the colorbar issue - directly use scatter for the colorbar
        scatter = plt.scatter(
            comp["weight_decay"],
            comp["micro_f1"],
            c=comp["warmup_ratio"],
            s=comp["micro_f1"] * 500,  # Size proportional to micro_f1
            cmap="viridis",
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add trial labels
        for i, row in comp.iterrows():
            plt.annotate(
                f"#{int(row['trial'])}", 
                (row["weight_decay"], row["micro_f1"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha='center', 
                fontsize=9,
                fontweight='semibold'
            )
            
        # Add colorbar using the scatter object
        cbar = plt.colorbar(scatter)
        cbar.set_label('Warmup Ratio', fontsize=11)
            
        plt.title("Impact of Weight Decay on Performance", fontweight='bold')
        plt.xlabel("Weight Decay")
        plt.ylabel("Micro F1 Score")
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Mark the best trial
        best_row = comp.loc[comp['micro_f1'].idxmax()]
        plt.scatter(
            best_row['weight_decay'], 
            best_row['micro_f1'],
            s=200, 
            facecolors='none', 
            edgecolors='red', 
            linewidth=2,
            zorder=10
        )
        
        plt.tight_layout()
        plt.savefig(figs_dir / "wd_vs_f1.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {figs_dir / 'wd_vs_f1.png'}")
    
    # 6. Time vs micro_f1 (two panels)
    if comp["train_seconds"].notna().any():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
        
        # Create discrete color map for epochs
        epoch_values = sorted(comp["epochs"].unique())
        # Fix the deprecated get_cmap warning
        colors = sns.color_palette("viridis", len(epoch_values))
        colors_dict = {val: colors[i] for i, val in enumerate(epoch_values)}
        
        # Left subplot - Training Time
        for epoch_val in epoch_values:
            epoch_df = comp[comp["epochs"] == epoch_val]
            ax1.scatter(
                epoch_df["train_seconds"],
                epoch_df["micro_f1"],
                s=120,
                color=colors_dict[epoch_val],
                label=f"{int(epoch_val)} epochs",
                alpha=0.8,
                edgecolor='white',
                linewidth=0.5
            )
            
            # Add trial labels
            for i, row in epoch_df.iterrows():
                ax1.annotate(
                    f"#{int(row['trial'])}", 
                    (row["train_seconds"], row["micro_f1"]),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha='center', 
                    fontsize=9,
                    fontweight='semibold'
                )
        
        ax1.set_title("Training Time vs Performance", fontweight='bold')
        ax1.set_xlabel("Training Time (seconds)")
        ax1.set_ylabel("Micro F1 Score")
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Better legend formatting
        legend1 = ax1.legend(frameon=True, fontsize=9)
        legend1.get_frame().set_alpha(0.9)
        legend1.get_frame().set_edgecolor('lightgray')
        
        # Right subplot - Generation Time (if available)
        if comp["gen_seconds"].notna().any():
            for epoch_val in epoch_values:
                epoch_df = comp[comp["epochs"] == epoch_val]
                ax2.scatter(
                    epoch_df["gen_seconds"],
                    epoch_df["micro_f1"],
                    s=120,
                    color=colors_dict[epoch_val],
                    label=f"{int(epoch_val)} epochs",
                    alpha=0.8,
                    edgecolor='white',
                    linewidth=0.5
                )
                
                # Add trial labels
                for i, row in epoch_df.iterrows():
                    ax2.annotate(
                        f"#{int(row['trial'])}", 
                        (row["gen_seconds"], row["micro_f1"]),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha='center', 
                        fontsize=9,
                        fontweight='semibold'
                    )
            
            ax2.set_title("Generation Time vs Performance", fontweight='bold')
            ax2.set_xlabel("Generation Time (seconds)")
            ax2.set_ylabel("Micro F1 Score")
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            # No need for second legend, it would be identical
        
        plt.suptitle("Time Efficiency Analysis", fontweight='bold', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for the suptitle
        plt.savefig(figs_dir / "time_vs_f1.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {figs_dir / 'time_vs_f1.png'}")
    
    # 7. Correlation heatmap
    num_cols = [
        "micro_f1", "epochs", "learning_rate", "warmup_ratio", "weight_decay",
        "grad_accum", "lora_r", "lora_alpha", "lora_dropout", "train_seconds", 
        "gen_seconds", "disk_mb"
    ]
    num_cols = [c for c in num_cols if c in comp.columns and comp[c].notna().any()]
    
    if len(num_cols) >= 2:
        try:
            # Check pandas version for correct corr method
            pd_version = pd.__version__
            major, minor = map(int, pd_version.split('.')[0:2])
            
            if major >= 1 and minor >= 5:
                # For pandas >= 1.5.0, use numeric_only
                corr = comp[num_cols].corr(numeric_only=True)
            else:
                # For older pandas versions
                corr = comp[num_cols].corr()
                
            plt.figure(figsize=(12, 10))
            
            # Create a nicer heatmap
            mask = np.zeros_like(corr, dtype=bool)
            mask[np.triu_indices_from(mask, 1)] = True  # Show only lower triangle
            
            # Better color map for correlation
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(
                corr, 
                annot=True, 
                fmt=".2f", 
                cmap=cmap,
                center=0,
                linewidths=.5, 
                cbar_kws={"shrink": .8},
                square=True,
                mask=mask,
                annot_kws={"size": 9}
            )
            plt.title("Parameter Correlation Matrix", fontweight='bold')
            plt.tight_layout()
            plt.savefig(figs_dir / "corr_heatmap.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {figs_dir / 'corr_heatmap.png'}")
        except Exception as e:
            print(f"Error generating correlation heatmap: {e}")
    
    # 8. Feature importance (using RandomForest)
    try:
        from sklearn.ensemble import RandomForestRegressor
        hp_cols = ["epochs", "learning_rate", "warmup_ratio", "weight_decay", 
                  "grad_accum", "lora_r", "lora_alpha", "lora_dropout", "patience"]
        hp_cols = [c for c in hp_cols if c in comp.columns and comp[c].notna().any()]
        feat_df = comp.dropna(subset=["micro_f1"] + hp_cols)
        
        if not feat_df.empty and len(hp_cols) >= 2 and feat_df["micro_f1"].nunique() > 1:
            X = feat_df[hp_cols].astype(float)
            y = feat_df["micro_f1"].astype(float)
            rf = RandomForestRegressor(n_estimators=300, random_state=42)
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=hp_cols).sort_values(ascending=False)
            
            plt.figure(figsize=(10, 7))
            
            # Create nicer bar chart
            bars = plt.barh(
                importances.index, 
                importances.values, 
                color=palette[0],
                alpha=0.8,
                edgecolor='white',
                linewidth=0.5
            )
            
            # Add value annotations
            for i, v in enumerate(importances.values):
                plt.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=9, fontweight='semibold')
                
            plt.title("Hyperparameter Importance Analysis", fontweight='bold')
            plt.xlabel("Relative Importance")
            plt.ylabel("Hyperparameter")
            plt.grid(axis='x', alpha=0.3, linestyle='--')
            plt.xlim(0, max(importances.values) * 1.15)  # Add space for annotations
                
            plt.tight_layout()
            plt.savefig(figs_dir / "hp_importance.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {figs_dir / 'hp_importance.png'}")
    except Exception as e:
        print(f"Error generating feature importance: {e}")
        
    # 9. Pruned trials: last eval loss distribution
    if not pruned.empty and pruned["last_eval_loss"].notna().any():
        plt.figure(figsize=(10, 6))
        
        # Create a nicer histogram
        sns.histplot(
            pruned["last_eval_loss"], 
            bins=min(20, len(pruned)),  # Adjust bins based on data size
            kde=True, 
            color=palette[1],
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add vertical line for mean
        mean_loss = pruned["last_eval_loss"].mean()
        plt.axvline(
            x=mean_loss, 
            color='darkred', 
            linestyle='--', 
            linewidth=1.5,
            alpha=0.7
        )
        plt.text(
            mean_loss, 
            plt.gca().get_ylim()[1] * 0.95, 
            f" Mean: {mean_loss:.4f}", 
            fontsize=9,
            fontweight='semibold',
            color='darkred',
            va='top'
        )
        
        plt.title("Evaluation Loss Distribution in Pruned Trials", fontweight='bold')
        plt.xlabel("Last Evaluation Loss")
        plt.ylabel("Count")
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(figs_dir / "pruned_eval_loss_hist.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {figs_dir / 'pruned_eval_loss_hist.png'}")

    # Create a README.md in the figs directory explaining the files
    readme_text = """# HPO Visualization Figures

These figures visualize the results of hyperparameter optimization trials.

## List of Files:
- `micro_f1_by_trial.png`: Bar chart showing micro_f1 metric for each completed trial
- `lr_vs_f1.png`: Scatter plot of learning rate vs micro_f1
- `epochs_vs_f1.png`: Scatter plot of epochs vs micro_f1
- `lora_params_vs_f1.png`: Two-panel plot showing LoRA parameters impact on micro_f1
- `wd_vs_f1.png`: Scatter plot of weight decay vs micro_f1
- `time_vs_f1.png`: Two-panel plot showing training and generation time impact
- `corr_heatmap.png`: Correlation heatmap between hyperparameters and micro_f1
- `hp_importance.png`: Random Forest feature importance for hyperparameters
- `pruned_eval_loss_hist.png`: Distribution of evaluation loss for pruned trials
"""
    (figs_dir / "README.md").write_text(readme_text, encoding="utf-8")


# ---------------------------- report ----------------------------

def make_text_report(df: pd.DataFrame, parsed: Dict[str, Any]) -> str:
    comp = df[df["state"] == "COMPLETED"].copy()
    pruned = df[df["state"] == "PRUNED"].copy()

    lines = []
    lines.append("# HPO Log Report")
    lines.append("")
    lines.append(f"- Completed trials: {len(comp)}")
    lines.append(f"- Pruned trials: {len(pruned)}")
    lines.append(f"- Unknown: {int((df['state'] == 'UNKNOWN').sum())}")
    lines.append("")

    if not comp.empty:
        top = comp.sort_values("micro_f1", ascending=False).head(5)
        lines.append("## Top trials (by micro_f1)")
        for _, r in top.iterrows():
            lines.append(
                f"- Trial {int(r['trial'])}: micro_f1={r['micro_f1']:.4f}, "
                f"epochs={r.get('epochs')}, lr={r.get('learning_rate')}, "
                f"warmup={r.get('warmup_ratio')}, wd={r.get('weight_decay')}, "
                f"grad_accum={r.get('grad_accum')}, lora_r={r.get('lora_r')}, "
                f"lora_alpha={r.get('lora_alpha')}, dropout={r.get('lora_dropout')}, "
                f"train_s={r.get('train_seconds')}, gen_s={r.get('gen_seconds')}"
            )

        # Pearson correlations
        num_cols = [
            "epochs","learning_rate","warmup_ratio","weight_decay","grad_accum",
            "lora_r","lora_alpha","lora_dropout","train_seconds","gen_seconds","disk_mb"
        ]
        cols = [c for c in num_cols if c in comp.columns]
        try:
            # Check pandas version for correct corr method
            pd_version = pd.__version__
            major, minor = map(int, pd_version.split('.')[0:2])
            
            if major >= 1 and minor >= 5:
                # For pandas >= 1.5.0, use numeric_only
                corr = comp[cols + ["micro_f1"]].corr(numeric_only=True)["micro_f1"].drop(labels=["micro_f1"])
            else:
                # For older pandas versions, filter columns first
                num_df = comp[cols + ["micro_f1"]].select_dtypes(include=['number'])
                corr = num_df.corr()["micro_f1"].drop(labels=["micro_f1"])
                
            lines.append("")
            lines.append("## Pearson correlation with micro_f1 (completed trials)")
            for k, v in corr.sort_values(ascending=False).items():
                if not math.isnan(v):
                    lines.append(f"- {k}: {v:.3f}")
        except Exception as e:
            lines.append(f"\nError calculating correlations: {e}")

    best = parsed.get("best", {})
    if best and best.get("trial") is not None:
        lines.append("")
        lines.append("## Best (from log tail)")
        lines.append(f"- Trial: {best.get('trial')}")
        if best.get("metric_value") is not None:
            lines.append(f"- micro_f1: {best.get('metric_value'):.4f}")
        if isinstance(best.get("params"), dict):
            bp = best.get("params")
            pretty = ", ".join([f"{k}={bp[k]}" for k in sorted(bp.keys())])
            lines.append(f"- Params: {pretty}")

    return "\n".join(lines)


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_file", required=True, help="Path to HPO console log file")
    ap.add_argument("--out_dir", required=True, help="Directory to write analysis outputs")
    args = ap.parse_args()

    log_path = Path(args.log_file)
    out_dir = Path(args.out_dir)
    figs_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Read log file
    print(f"Reading log file: {log_path}")
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Error reading log file: {e}")
        return 1
        
    # Parse and convert to DataFrame
    print("Parsing log data...")
    parsed = parse_log_text(text)
    df = parsed_to_df(parsed)

    # Save table & summary
    print(f"Saving trial data to: {out_dir / 'trials.csv'}")
    df.to_csv(out_dir / "trials.csv", index=False)

    summary = {
        "completed_trials": int((df["state"] == "COMPLETED").sum()),
        "pruned_trials": int((df["state"] == "PRUNED").sum()),
        "unknown_trials": int((df["state"] == "UNKNOWN").sum()),
        "best_from_tail": parsed.get("best", {}),
        "max_micro_f1": float(df["micro_f1"].max()) if df["micro_f1"].notna().any() else None,
        "best_trial_by_row": int(df.loc[df["micro_f1"].idxmax(), "trial"]) if df["micro_f1"].notna().any() else None,
    }
    
    print(f"Saving summary to: {out_dir / 'summary.json'}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Text report
    print("Generating text report...")
    report_md = make_text_report(df, parsed)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    # Generate matplotlib figures
    print("Creating visualization figures in:", figs_dir)
    make_matplotlib_figures(df, figs_dir)
    
    print(f"\n[SUCCESS] Analysis completed!")
    print(f"Parsed {len(df)} trials. Completed={summary['completed_trials']} Pruned={summary['pruned_trials']}")
    if summary["max_micro_f1"] is not None:
        print(f"Best micro_f1 by table: {summary['max_micro_f1']:.4f} (trial {summary['best_trial_by_row']})")
    bf = parsed.get("best", {})
    if bf.get("trial") is not None:
        print(f"Best (tail): trial {bf['trial']} with micro_f1={bf.get('metric_value')}")
    print(f"Outputs written to: {out_dir}")
    print(f"Figures saved in: {figs_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())