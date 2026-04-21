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

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOGS_DIR / f"platform_sector_summary_{TIMESTAMP}.log"

# =========================================================
# IMPORTANT:
# False = save graphs with REAL values
# True  = normalize graph bars to 0..1 for visual comparison
#
# Recommended: keep False unless you explicitly want normalized graphs
# =========================================================
NORMALIZE_PLOTS = False


# =========================
# LOGGING
# =========================
def log(msg):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
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


def coerce_success(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

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
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(success_map)
    )
    fallback = pd.to_numeric(series, errors="coerce")
    return mapped.where(mapped.notna(), fallback)


def clean_string_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df

    df = df.copy()
    df[col] = df[col].astype(str).str.strip()
    df = df[df[col] != ""]
    df = df[df[col].str.lower() != "nan"]
    return df


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        for col in df.columns.to_flat_index()
    ]
    return df


# =========================
# NORMALIZATION
# =========================
def minmax(series: pd.Series) -> pd.Series:
    """
    Normalize values to 0..1 for plotting only.
    Does NOT change CSV output values.
    """
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


# =========================
# AGGREGATION
# =========================
def build_aggregation(
    df: pd.DataFrame,
    group_col: str,
    col_success: str,
    col_time: str,
    col_cog: str,
    col_cog_peak: str,
):
    agg = {}

    if col_success and col_success in df.columns:
        agg[col_success] = ["count", "mean"]

    if "Participant" in df.columns:
        agg["Participant"] = [pd.Series.nunique]

    if "Task" in df.columns:
        agg["Task"] = [pd.Series.nunique]

    for col in [col_time, col_cog, col_cog_peak]:
        if col and col in df.columns:
            agg[col] = ["mean", "median", "std"]

    if not agg:
        raise ValueError("No valid aggregation columns found")

    out = df.groupby(group_col, dropna=False).agg(agg).reset_index()
    out = flatten_columns(out)

    rename_map = {
        group_col: group_col,
        "Participant_nunique": "ParticipantCount",
        "Task_nunique": "TaskCount",
    }

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

    out = out.rename(columns=rename_map)

    preferred_order = [
        group_col,
        "ParticipantCount",
        "TaskCount",
        "N",
        "SuccessRate",
        "MeanTime",
        "MedianTime",
        "StdTime",
        "MeanCognitiveLoad",
        "MedianCognitiveLoad",
        "StdCognitiveLoad",
        "MeanCognitivePeak",
        "MedianCognitivePeak",
        "StdCognitivePeak",
    ]
    cols_present = [c for c in preferred_order if c in out.columns]
    other_cols = [c for c in out.columns if c not in cols_present]
    out = out[cols_present + other_cols]

    out[group_col] = out[group_col].astype(str)
    out = out.sort_values(group_col).reset_index(drop=True)

    return out


# =========================
# PLOTTING
# =========================
def save_bar(df, x_col, y_col, title, ylabel, outpath, normalize=False):
    if y_col not in df.columns:
        log(f"[WARN] Plot skipped, missing column: {y_col}")
        return

    plot_df = df[[x_col, y_col]].copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[y_col])

    if plot_df.empty:
        log(f"[WARN] Plot skipped, no valid data for: {y_col}")
        return

    plot_df = plot_df.sort_values(y_col, ascending=False).reset_index(drop=True)

    raw_values = plot_df[y_col].copy()
    plot_values = plot_df[y_col].copy()

    if normalize:
        plot_values = minmax(plot_values)
        ylabel = f"{ylabel} (normalized 0-1)"

    plt.figure(figsize=(10, 4))
    bars = plt.bar(plot_df[x_col].astype(str), plot_values)

    for bar, raw_val in zip(bars, raw_values):
        if pd.notna(raw_val):
            height = bar.get_height()
            y_text = height if pd.notna(height) else 0
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                y_text,
                f"{raw_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

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
def main():
    log("=== PLATFORM / SECTOR SUMMARY SCRIPT STARTED ===")
    log(f"Working directory: {Path.cwd()}")
    log(f"Looking for master file: {MASTER_FILE.resolve()}")

    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_FILE.resolve()}")

    df = pd.read_csv(MASTER_FILE)
    log(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

    cols = list(df.columns)
    log("Columns found:")
    for c in cols:
        log(f" - {c}")

    if "Platform" not in cols:
        raise ValueError("Column 'Platform' is missing")
    if "Sector" not in cols:
        raise ValueError("Column 'Sector' is missing")

    # defensive column matching
    col_success = find_column(cols, ["Success"])
    col_time = find_column(cols, ["CompletionTime_sec", "CompletionTimeMean_sec", "CompletionTime"])
    col_cog = find_column(cols, ["CogLoadMean", "MeanCognitiveLoad", "Cognitive Load"])
    col_cog_peak = find_column(cols, ["CogLoadPeak", "CogLoadPeakMean", "MeanCognitivePeak"])

    log("Selected columns:")
    log(f" Success: {col_success}")
    log(f" Time: {col_time}")
    log(f" Cognitive: {col_cog}")
    log(f" Cognitive Peak: {col_cog_peak}")

    numeric_cols = [col_time, col_cog, col_cog_peak]
    df = coerce_numeric(df, numeric_cols)

    if col_success and col_success in df.columns:
        df[col_success] = coerce_success(df[col_success])

    df = clean_string_column(df, "Platform")
    df = clean_string_column(df, "Sector")

    if "Participant" in df.columns:
        df = clean_string_column(df, "Participant")
    if "Task" in df.columns:
        df = clean_string_column(df, "Task")

    if df.empty:
        raise ValueError("Dataset is empty after cleaning")

    # =========================
    # BUILD SUMMARIES
    # =========================
    platform_summary = build_aggregation(
        df=df,
        group_col="Platform",
        col_success=col_success,
        col_time=col_time,
        col_cog=col_cog,
        col_cog_peak=col_cog_peak,
    )

    sector_summary = build_aggregation(
        df=df,
        group_col="Sector",
        col_success=col_success,
        col_time=col_time,
        col_cog=col_cog,
        col_cog_peak=col_cog_peak,
    )

    platform_summary_file = SUMMARIES_DIR / "platform_summary.csv"
    sector_summary_file = SUMMARIES_DIR / "sector_summary.csv"

    platform_summary.to_csv(platform_summary_file, index=False)
    sector_summary.to_csv(sector_summary_file, index=False)

    log(f"[OK] Saved CSV: {platform_summary_file.resolve()}")
    log(f"[OK] Saved CSV: {sector_summary_file.resolve()}")

    # =========================
    # SAVE FIGURES - PLATFORM
    # =========================
    save_bar(
        platform_summary,
        "Platform",
        "SuccessRate",
        "Success Rate by Platform",
        "Success Rate",
        FIGURES_DIR / "platform_success_rate.png",
        normalize=NORMALIZE_PLOTS,
    )

    save_bar(
        platform_summary,
        "Platform",
        "MeanTime",
        "Mean Completion Time by Platform",
        "Completion Time (sec)",
        FIGURES_DIR / "platform_completion_time.png",
        normalize=NORMALIZE_PLOTS,
    )

    save_bar(
        platform_summary,
        "Platform",
        "MeanCognitiveLoad",
        "Mean Cognitive Load by Platform",
        "Cognitive Load",
        FIGURES_DIR / "platform_cognitive_load.png",
        normalize=NORMALIZE_PLOTS,
    )

    # =========================
    # SAVE FIGURES - SECTOR
    # =========================
    save_bar(
        sector_summary,
        "Sector",
        "SuccessRate",
        "Success Rate by Sector",
        "Success Rate",
        FIGURES_DIR / "sector_success_rate.png",
        normalize=NORMALIZE_PLOTS,
    )

    save_bar(
        sector_summary,
        "Sector",
        "MeanTime",
        "Mean Completion Time by Sector",
        "Completion Time (sec)",
        FIGURES_DIR / "sector_completion_time.png",
        normalize=NORMALIZE_PLOTS,
    )

    save_bar(
        sector_summary,
        "Sector",
        "MeanCognitiveLoad",
        "Mean Cognitive Load by Sector",
        "Cognitive Load",
        FIGURES_DIR / "sector_cognitive_load.png",
        normalize=NORMALIZE_PLOTS,
    )

    log("")
    log("Platform summary preview:")
    log(platform_summary.to_string(index=False))

    log("")
    log("Sector summary preview:")
    log(sector_summary.to_string(index=False))

    log("Finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("=== ERROR ===")
        log(f"Error: {e}")
        log(traceback.format_exc())
    finally:
        log(f"Log file: {LOG_PATH.resolve()}")