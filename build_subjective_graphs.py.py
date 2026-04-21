from pathlib import Path
from datetime import datetime
import traceback
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# PATHS
# =========================
BASE_DIR = Path(r"C:\Users\lakis\Desktop\UX_Analysis_Tool")

DATA_DIR = BASE_DIR / "data" / "zSUS_NASA-TLX"
FIGURES_DIR = BASE_DIR / "results" / "figures"
SUMMARY_DIR = BASE_DIR / "results" / "summaries"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = FIGURES_DIR / f"subjective_graphs_{TIMESTAMP}.log"

# Your cleaned CSVs use comma separator and UTF-8 BOM
CSV_SEPARATOR = ","
CSV_ENCODING = "utf-8-sig"


# =========================
# LOGGING
# =========================
def log(msg: str):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


# =========================
# HELPERS
# =========================
def sus_grade(score):
    if pd.isna(score):
        return ""
    if score >= 80:
        return "A"
    elif score >= 68:
        return "B"
    elif score >= 50:
        return "C"
    else:
        return "D"


def tlx_level(score):
    if pd.isna(score):
        return ""
    if score < 33:
        return "Low"
    elif score <= 66:
        return "Moderate"
    else:
        return "High"


def find_csv_file(folder: Path, keyword: str) -> Path:
    matches = sorted(folder.glob(f"*{keyword}*.csv"))
    if not matches:
        raise FileNotFoundError(f"No CSV file found with keyword '{keyword}' in: {folder}")
    if len(matches) > 1:
        log(f"[WARN] Multiple files matched '{keyword}'. Using first one:")
        for m in matches:
            log(f"       - {m.name}")
    return matches[0]


def clean_platform_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )


def save_bar_chart(df, x_col, y_col, title, ylabel, filename):
    plot_df = df[[x_col, y_col]].copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df[x_col] = plot_df[x_col].astype(str).str.strip()
    plot_df = plot_df.dropna(subset=[y_col]).sort_values(y_col, ascending=False)

    if plot_df.empty:
        log(f"[WARN] No valid data for bar chart: {filename}")
        return

    plt.figure(figsize=(10, 5))
    bars = plt.bar(plot_df[x_col], plot_df[y_col])

    for bar, val in zip(bars, plot_df[y_col]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    outpath = FIGURES_DIR / filename
    plt.savefig(outpath, dpi=300)
    plt.close()
    log(f"[OK] Saved graph: {outpath}")


def save_scatter_chart(df, x_col, y_col, label_col, title, xlabel, ylabel, filename):
    plot_df = df[[x_col, y_col, label_col]].copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df[label_col] = plot_df[label_col].astype(str).str.strip()
    plot_df = plot_df.dropna(subset=[x_col, y_col])

    if plot_df.empty:
        log(f"[WARN] No valid data for scatter chart: {filename}")
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(plot_df[x_col], plot_df[y_col])

    for _, row in plot_df.iterrows():
        plt.text(
            row[x_col] + 0.8,
            row[y_col] + 0.8,
            str(row[label_col]),
            fontsize=9,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.tight_layout()

    outpath = FIGURES_DIR / filename
    plt.savefig(outpath, dpi=300)
    plt.close()
    log(f"[OK] Saved graph: {outpath}")


# =========================
# NASA-TLX
# =========================
def compute_nasa_summary(nasa_file: Path) -> pd.DataFrame:
    log(f"[INFO] Reading NASA-TLX file: {nasa_file}")
    df = pd.read_csv(nasa_file, sep=CSV_SEPARATOR, encoding=CSV_ENCODING)

    # Clean headers
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    log(f"[INFO] NASA rows: {len(df)}, columns: {len(df.columns)}")
    log("[INFO] NASA columns:")
    for c in df.columns:
        log(f" - {c}")

    platform_col = df.columns[0]
    df[platform_col] = clean_platform_series(df[platform_col])

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 6:
        raise ValueError(
            f"NASA file should contain at least 6 numeric workload columns, but found {len(numeric_cols)}."
        )

    tlx_cols = [c for c in numeric_cols if not str(c).lower().startswith("unnamed")]

    if len(tlx_cols) < 6:
        raise ValueError(
            f"NASA usable numeric workload columns are too few after cleaning: {len(tlx_cols)}."
        )

    log("[INFO] NASA numeric columns used for TLX calculation:")
    for c in tlx_cols:
        log(f" - {c}")

    # IMPORTANT: your NASA file uses a 1-10 scale
    df["TLX_raw"] = df[tlx_cols].mean(axis=1)
    df["TLX_100"] = (df["TLX_raw"] / 10.0) * 100.0

    summary = (
        df.groupby(platform_col, dropna=False)
        .agg(
            Mean_TLX=("TLX_100", "mean"),
            Std_TLX=("TLX_100", "std"),
            N=("TLX_100", "count"),
        )
        .reset_index()
        .rename(columns={platform_col: "Platform"})
    )

    summary["Platform"] = clean_platform_series(summary["Platform"])
    summary["WorkloadLevel"] = summary["Mean_TLX"].apply(tlx_level)
    summary = summary.sort_values("Mean_TLX", ascending=False).reset_index(drop=True)

    return summary


# =========================
# SUS
# =========================
def compute_sus_summary(sus_file: Path) -> pd.DataFrame:
    log(f"[INFO] Reading SUS file: {sus_file}")
    df = pd.read_csv(sus_file, sep=CSV_SEPARATOR, encoding=CSV_ENCODING)

    # Clean headers
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    log(f"[INFO] SUS rows: {len(df)}, columns: {len(df.columns)}")
    log("[INFO] SUS columns:")
    for c in df.columns:
        log(f" - {c}")

    platform_col = df.columns[0]
    df[platform_col] = clean_platform_series(df[platform_col])

    sus_cols = df.select_dtypes(include="number").columns.tolist()

    if len(sus_cols) != 10:
        raise ValueError(
            f"SUS file should contain 10 numeric question columns, but found {len(sus_cols)}."
        )

    log("[INFO] SUS numeric columns used:")
    for c in sus_cols:
        log(f" - {c}")

    def compute_sus(row):
        total = 0.0
        for i, col in enumerate(sus_cols):
            val = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(val):
                return pd.NA
            if i % 2 == 0:  # Q1, Q3, Q5, Q7, Q9
                total += (val - 1)
            else:  # Q2, Q4, Q6, Q8, Q10
                total += (5 - val)
        return total * 2.5

    # Learnability = items 4 and 10
    learn_cols = [sus_cols[3], sus_cols[9]]

    def compute_learnability(row):
        q4 = pd.to_numeric(row[learn_cols[0]], errors="coerce")
        q10 = pd.to_numeric(row[learn_cols[1]], errors="coerce")
        if pd.isna(q4) or pd.isna(q10):
            return pd.NA
        return ((5 - q4) + (5 - q10)) * 12.5

    def compute_usability(row):
        total = 0.0
        for i, col in enumerate(sus_cols):
            if col in learn_cols:
                continue
            val = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(val):
                return pd.NA
            if i % 2 == 0:
                total += (val - 1)
            else:
                total += (5 - val)
        return total * (100.0 / 32.0)

    df["SUS_Score"] = df.apply(compute_sus, axis=1)
    df["Learnability"] = df.apply(compute_learnability, axis=1)
    df["Usability"] = df.apply(compute_usability, axis=1)

    summary = (
        df.groupby(platform_col, dropna=False)
        .agg(
            SUS_Score=("SUS_Score", "mean"),
            Learnability=("Learnability", "mean"),
            Usability=("Usability", "mean"),
            N=("SUS_Score", "count"),
        )
        .reset_index()
        .rename(columns={platform_col: "Platform"})
    )

    summary["Platform"] = clean_platform_series(summary["Platform"])
    summary["Grade"] = summary["SUS_Score"].apply(sus_grade)
    summary = summary.sort_values("SUS_Score", ascending=False).reset_index(drop=True)

    return summary


# =========================
# MAIN
# =========================
def main():
    log("=== SUBJECTIVE GRAPHS SCRIPT STARTED ===")
    log(f"BASE_DIR: {BASE_DIR}")
    log(f"DATA_DIR: {DATA_DIR}")
    log(f"FIGURES_DIR: {FIGURES_DIR}")
    log(f"SUMMARY_DIR: {SUMMARY_DIR}")
    log(f"LOG_FILE: {LOG_FILE}")
    log(f"CSV_SEPARATOR: {CSV_SEPARATOR}")
    log(f"CSV_ENCODING: {CSV_ENCODING}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    nasa_file = find_csv_file(DATA_DIR, "nasa")
    sus_file = find_csv_file(DATA_DIR, "sus")

    log(f"[INFO] Using NASA file: {nasa_file}")
    log(f"[INFO] Using SUS file: {sus_file}")

    nasa_summary = compute_nasa_summary(nasa_file)
    sus_summary = compute_sus_summary(sus_file)

    # Save summaries to results/summaries
    nasa_csv = SUMMARY_DIR / "nasa_tlx_summary.csv"
    sus_csv = SUMMARY_DIR / "sus_summary.csv"

    nasa_summary.to_csv(nasa_csv, index=False, encoding="utf-8-sig")
    sus_summary.to_csv(sus_csv, index=False, encoding="utf-8-sig")

    log(f"[OK] Saved summary CSV: {nasa_csv}")
    log(f"[OK] Saved summary CSV: {sus_csv}")

    # Save bar charts to results/figures
    save_bar_chart(
        df=nasa_summary,
        x_col="Platform",
        y_col="Mean_TLX",
        title="Mean NASA-TLX Scores by Platform",
        ylabel="NASA-TLX (0-100)",
        filename="nasa_tlx_by_platform.png",
    )

    save_bar_chart(
        df=sus_summary,
        x_col="Platform",
        y_col="SUS_Score",
        title="SUS Scores by Platform",
        ylabel="SUS Score (0-100)",
        filename="sus_by_platform.png",
    )

    # Normalize names before merge so Gov.gr / gov.gr etc. do not disappear
    nasa_merge = nasa_summary.copy()
    sus_merge = sus_summary.copy()

    nasa_merge["Platform"] = nasa_merge["Platform"].astype(str).str.strip().str.lower()
    sus_merge["Platform"] = sus_merge["Platform"].astype(str).str.strip().str.lower()

    merged = pd.merge(
        nasa_merge[["Platform", "Mean_TLX"]],
        sus_merge[["Platform", "SUS_Score"]],
        on="Platform",
        how="inner",
    )

    display_map = {
        "aade": "AADE",
        "efka": "EFKA",
        "gov.gr": "Gov.gr",
        "telecom": "Telecom",
        "myhealth": "MyHealth",
        "web banking": "Web Banking",
    }
    merged["Platform"] = merged["Platform"].map(display_map).fillna(merged["Platform"])

    log("[INFO] Merged NASA + SUS for scatter:")
    log(merged.round(2).to_string(index=False))

    save_scatter_chart(
        df=merged,
        x_col="Mean_TLX",
        y_col="SUS_Score",
        label_col="Platform",
        title="Relationship Between NASA-TLX and SUS Scores Across Platforms",
        xlabel="NASA-TLX (0-100)",
        ylabel="SUS Score (0-100)",
        filename="sus_vs_tlx_scatter.png",
    )

    log("")
    log("NASA summary preview:")
    log(nasa_summary.round(2).to_string(index=False))

    log("")
    log("SUS summary preview:")
    log(sus_summary.round(2).to_string(index=False))

    log("")
    log("Finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("=== ERROR ===")
        log(f"Error: {e}")
        log(traceback.format_exc())