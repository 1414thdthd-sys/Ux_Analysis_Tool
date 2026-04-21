import sys
from pathlib import Path
from datetime import datetime
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# PATHS
# =========================
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent.parent

MASTER_FILE = APP_DIR / "results" / "master" / "master_dataset.csv"
RESULTS_DIR = APP_DIR / "results"
SUMMARIES_DIR = RESULTS_DIR / "summaries"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOGS_DIR / f"task_summary_{timestamp}.log"


# =========================
# LOGGING
# =========================
def log(msg):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


# =========================
# HELPERS
# =========================
def find_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def coerce_numeric(df, columns_to_convert):
    for col in columns_to_convert:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def minmax(series):
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)

    valid = s.dropna()
    if valid.empty:
        return out

    mn = valid.min()
    mx = valid.max()

    if mn == mx:
        out.loc[s.notna()] = 1.0
        return out

    out.loc[s.notna()] = (s.loc[s.notna()] - mn) / (mx - mn)
    return out


def save_bar(df, x_col, y_col, title, ylabel, outpath):
    if y_col not in df.columns:
        log(f"[WARN] Plot skipped, missing column: {y_col}")
        return

    vals = pd.to_numeric(df[y_col], errors="coerce")
    plot_df = df[[x_col, y_col]].copy()
    plot_df[y_col] = vals
    plot_df = plot_df.dropna(subset=[y_col])

    if plot_df.empty:
        log(f"[WARN] Plot skipped, no valid data for: {y_col}")
        return

    plt.figure(figsize=(10, 4))
    plt.bar(plot_df[x_col].astype(str), plot_df[y_col])
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    log(f"[OK] Saved plot: {outpath}")


# =========================
# MAIN
# =========================
try:
    log("=== TASK SUMMARY SCRIPT STARTED ===")
    log(f"Working directory: {Path.cwd()}")
    log(f"Looking for master file: {MASTER_FILE.resolve()}")

    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_FILE.resolve()}")

    df = pd.read_csv(MASTER_FILE)
    log(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

    cols = list(df.columns)
    log("Columns:")
    for c in cols:
        log(f" - {c}")

    if "Task" not in cols:
        raise ValueError("Column 'Task' is missing")

    # Raw columns expected from merged master dataset
    col_success = find_column(cols, ["Success"])
    col_time = find_column(cols, ["CompletionTime_sec"])
    col_cog = find_column(cols, ["CogLoadMean"])
    col_cog_peak = find_column(cols, ["CogLoadPeak"])
    col_eda = find_column(cols, ["EDA_PeakRateMean", "EDA_TonicMean"])
    col_hr = find_column(cols, ["HR_Delta", "HR_Mean"])
    col_conf = find_column(cols, ["ConfusedMean"])
    col_fix = find_column(cols, ["EstimatedFixationEpisodes"])
    col_fixdur = find_column(cols, ["EstimatedFixationMeanDuration_ms"])
    col_gazevalid = find_column(cols, ["GazeValidRatio"])

    log("Selected columns:")
    log(f" Success: {col_success}")
    log(f" Time: {col_time}")
    log(f" Cognitive: {col_cog}")
    log(f" Cognitive Peak: {col_cog_peak}")
    log(f" EDA: {col_eda}")
    log(f" HR: {col_hr}")
    log(f" Confusion: {col_conf}")
    log(f" Fixation Count: {col_fix}")
    log(f" Fixation Duration: {col_fixdur}")
    log(f" Gaze Valid Ratio: {col_gazevalid}")

    numeric_cols = [
        col_time,
        col_cog,
        col_cog_peak,
        col_eda,
        col_hr,
        col_conf,
        col_fix,
        col_fixdur,
        col_gazevalid,
    ]
    df = coerce_numeric(df, numeric_cols)

    if col_success and col_success in df.columns:
        if df[col_success].dtype == object:
            success_map = {
                "true": 1,
                "false": 0,
                "yes": 1,
                "no": 0,
                "success": 1,
                "failed": 0,
                "completed": 1,
                "incomplete": 0,
                "1": 1,
                "0": 0,
            }
            mapped = (
                df[col_success]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(success_map)
            )
            fallback = pd.to_numeric(df[col_success], errors="coerce")
            df[col_success] = mapped.where(mapped.notna(), fallback)
        else:
            df[col_success] = pd.to_numeric(df[col_success], errors="coerce")

    df["Task"] = df["Task"].astype(str).str.strip()
    df = df[df["Task"] != ""]
    df = df[df["Task"].str.lower() != "nan"]

    if df.empty:
        raise ValueError("Dataset is empty after cleaning Task values")

    # =========================
    # SUMMARY
    # =========================
    agg = {}

    if col_success and col_success in df.columns:
        agg[col_success] = ["count", "mean"]

    for col in [col_time, col_cog, col_cog_peak, col_eda, col_hr, col_conf, col_fix, col_fixdur, col_gazevalid]:
        if col and col in df.columns:
            agg[col] = ["mean", "median", "std"]

    if not agg:
        raise ValueError("No valid metric columns found")

    summary = df.groupby("Task", dropna=False).agg(agg)
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    rename_map = {}

    if col_success:
        rename_map[f"{col_success}_count"] = "N"
        rename_map[f"{col_success}_mean"] = "SuccessRate"

    if col_time:
        rename_map[f"{col_time}_mean"] = "MeanTime"
        rename_map[f"{col_time}_median"] = "MedianTime"
        rename_map[f"{col_time}_std"] = "StdTime"

    if col_cog:
        rename_map[f"{col_cog}_mean"] = "MeanCognitiveLoad"
        rename_map[f"{col_cog}_median"] = "MedianCognitiveLoad"
        rename_map[f"{col_cog}_std"] = "StdCognitiveLoad"

    if col_cog_peak:
        rename_map[f"{col_cog_peak}_mean"] = "MeanCognitivePeak"
        rename_map[f"{col_cog_peak}_median"] = "MedianCognitivePeak"
        rename_map[f"{col_cog_peak}_std"] = "StdCognitivePeak"

    if col_eda:
        rename_map[f"{col_eda}_mean"] = "MeanEDA"
        rename_map[f"{col_eda}_median"] = "MedianEDA"
        rename_map[f"{col_eda}_std"] = "StdEDA"

    if col_hr:
        rename_map[f"{col_hr}_mean"] = "MeanHR"
        rename_map[f"{col_hr}_median"] = "MedianHR"
        rename_map[f"{col_hr}_std"] = "StdHR"

    if col_conf:
        rename_map[f"{col_conf}_mean"] = "MeanConfusion"
        rename_map[f"{col_conf}_median"] = "MedianConfusion"
        rename_map[f"{col_conf}_std"] = "StdConfusion"

    if col_fix:
        rename_map[f"{col_fix}_mean"] = "MeanFixationCount"
        rename_map[f"{col_fix}_median"] = "MedianFixationCount"
        rename_map[f"{col_fix}_std"] = "StdFixationCount"

    if col_fixdur:
        rename_map[f"{col_fixdur}_mean"] = "MeanFixationDuration"
        rename_map[f"{col_fixdur}_median"] = "MedianFixationDuration"
        rename_map[f"{col_fixdur}_std"] = "StdFixationDuration"

    if col_gazevalid:
        rename_map[f"{col_gazevalid}_mean"] = "MeanGazeValidRatio"
        rename_map[f"{col_gazevalid}_median"] = "MedianGazeValidRatio"
        rename_map[f"{col_gazevalid}_std"] = "StdGazeValidRatio"

    summary = summary.rename(columns=rename_map)

    summary_file = SUMMARIES_DIR / "task_summary.csv"
    summary.to_csv(summary_file, index=False)
    log(f"Saved summary to: {summary_file.resolve()}")

    # =========================
    # RANKINGS / COMPOSITE METRICS
    # =========================
    rank = summary.copy()

    if "MeanTime" in rank.columns:
        rank["Hardness"] = pd.to_numeric(rank["MeanTime"], errors="coerce")
    else:
        rank["Hardness"] = np.nan

    if "MeanCognitiveLoad" in rank.columns:
        rank["CognitiveDemand"] = pd.to_numeric(rank["MeanCognitiveLoad"], errors="coerce")
    else:
        rank["CognitiveDemand"] = np.nan

    if "MeanEDA" in rank.columns:
        rank["Stress"] = pd.to_numeric(rank["MeanEDA"], errors="coerce")
    elif "MeanHR" in rank.columns:
        rank["Stress"] = pd.to_numeric(rank["MeanHR"], errors="coerce")
    else:
        rank["Stress"] = np.nan

    if "MeanConfusion" in rank.columns:
        rank["ConfusionScore"] = pd.to_numeric(rank["MeanConfusion"], errors="coerce")
    else:
        rank["ConfusionScore"] = np.nan

    rank["Score_Success"] = minmax(rank["SuccessRate"]) if "SuccessRate" in rank.columns else np.nan
    rank["Score_Time"] = 1 - minmax(rank["MeanTime"]) if "MeanTime" in rank.columns else np.nan
    rank["Score_Cognitive"] = 1 - minmax(rank["MeanCognitiveLoad"]) if "MeanCognitiveLoad" in rank.columns else np.nan
    rank["Score_Stress"] = 1 - minmax(rank["Stress"]) if "Stress" in rank.columns else np.nan
    rank["Score_Confusion"] = 1 - minmax(rank["MeanConfusion"]) if "MeanConfusion" in rank.columns else np.nan

    score_cols = [
        "Score_Success",
        "Score_Time",
        "Score_Cognitive",
        "Score_Stress",
        "Score_Confusion",
    ]

    available_score_cols = [c for c in score_cols if c in rank.columns]
    weight_map = {
        "Score_Success": 0.35,
        "Score_Time": 0.25,
        "Score_Cognitive": 0.15,
        "Score_Stress": 0.15,
        "Score_Confusion": 0.10,
    }

    def weighted_row_score(row):
        numerator = 0.0
        denominator = 0.0
        for c in available_score_cols:
            val = row.get(c, np.nan)
            w = weight_map[c]
            if pd.notna(val):
                numerator += val * w
                denominator += w
        if denominator == 0:
            return np.nan
        return numerator / denominator

    rank["OverallScore"] = rank.apply(weighted_row_score, axis=1)
    rank = rank.sort_values("OverallScore", ascending=True, na_position="last").reset_index(drop=True)

    ranking_file = SUMMARIES_DIR / "task_rankings.csv"
    rank.to_csv(ranking_file, index=False)
    log(f"Saved rankings to: {ranking_file.resolve()}")

    # =========================
    # CHAPTER 4.4.3 PLOTS
    # =========================
    if "MeanTime" in summary.columns:
        save_bar(
            summary.sort_values("MeanTime", ascending=False),
            "Task",
            "MeanTime",
            "Mean Completion Time Across Tasks",
            "Completion Time (sec)",
            FIGURES_DIR / "mean_completion_time_across_tasks.png",
        )

    if "SuccessRate" in summary.columns:
        save_bar(
            summary.sort_values("SuccessRate", ascending=False),
            "Task",
            "SuccessRate",
            "Success Rate Across Tasks",
            "Success Rate",
            FIGURES_DIR / "success_rate_across_tasks.png",
        )

    if "MeanCognitiveLoad" in summary.columns:
        save_bar(
            summary.sort_values("MeanCognitiveLoad", ascending=False),
            "Task",
            "MeanCognitiveLoad",
            "Mean Cognitive Load Across Tasks",
            "Cognitive Load",
            FIGURES_DIR / "mean_cognitive_load_across_tasks.png",
        )

    if "MeanHR" in summary.columns:
        save_bar(
            summary.sort_values("MeanHR", ascending=False),
            "Task",
            "MeanHR",
            "Mean Heart Rate / HR Delta Across Tasks",
            "Heart Rate / HR Delta",
            FIGURES_DIR / "mean_hr_across_tasks.png",
        )

    if "MeanEDA" in summary.columns:
        save_bar(
            summary.sort_values("MeanEDA", ascending=False),
            "Task",
            "MeanEDA",
            "Mean EDA Across Tasks",
            "EDA",
            FIGURES_DIR / "mean_eda_across_tasks.png",
        )

    # =========================
    # RANKING / VARIABILITY PLOTS
    # =========================
    save_bar(
        rank.sort_values("OverallScore", ascending=True),
        "Task",
        "OverallScore",
        "Overall Score Ranking",
        "Overall Score",
        FIGURES_DIR / "overallscore_ranking.png",
    )

    if "StdTime" in rank.columns:
        save_bar(
            rank.sort_values("StdTime", ascending=False),
            "Task",
            "StdTime",
            "Completion Time Variability",
            "Std. Dev. of Completion Time",
            FIGURES_DIR / "variability_time.png",
        )

    if "StdCognitiveLoad" in rank.columns:
        save_bar(
            rank.sort_values("StdCognitiveLoad", ascending=False),
            "Task",
            "StdCognitiveLoad",
            "Cognitive Load Variability",
            "Std. Dev. of Cognitive Load",
            FIGURES_DIR / "variability_cognitiveload.png",
        )

    log("\nTop tasks:")
    log(rank.head(10).to_string(index=False))
    log("Finished.")

except Exception as e:
    log("=== ERROR ===")
    log(f"Error: {e}")
    log(traceback.format_exc())

finally:
    log(f"Log file: {log_path.resolve()}")