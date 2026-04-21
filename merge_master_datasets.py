import sys
import traceback
from datetime import datetime
from pathlib import Path
import pandas as pd

if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = APP_DIR / "data"
RESULTS_DIR = APP_DIR / "results"
MASTER_DIR = RESULTS_DIR / "master"
SUMMARIES_DIR = RESULTS_DIR / "summaries"
LOGS_DIR = RESULTS_DIR / "logs"

SECTOR_MAP = {
    "govgr": "Public",
    "aade": "Public",
    "efka": "Public",
    "health": "Public",
    "myhealth": "Public",
    "telecoms": "Private",
    "webbanking": "Private",
    "vodafone": "Private",
}

COLUMN_RENAME_MAP = {
    "participant": "Participant",
    "platform": "Platform",
    "sector": "Sector",
    "task": "Task",
    "status": "Status",
    "success": "Success",
}

CORE_COLUMNS = ["Participant", "Platform", "Sector", "Task", "Status", "Success"]


class TeeLogger:
    def __init__(self, logfile: Path):
        self.logfile = logfile
        self.logfile.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.logfile.open("a", encoding="utf-8")

    def write(self, message: str):
        print(message)
        self.fh.write(str(message) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def find_master_files(data_dir: Path, logger: TeeLogger):
    found = []

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")

    platform_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    logger.write(f"Found {len(platform_dirs)} platform folders under: {data_dir.resolve()}")

    for platform_dir in platform_dirs:
        platform = platform_dir.name
        analysis_dir = platform_dir / "analysis_output"

        if not analysis_dir.exists():
            logger.write(f"[WARN] Missing analysis_output folder for platform: {platform}")
            continue

        matches = sorted(list(analysis_dir.glob("*_master_dataset.csv")))

        if not matches:
            matches = sorted(list(analysis_dir.glob("*master*.csv")))

        if not matches:
            exact = analysis_dir / "master_dataset.csv"
            if exact.exists():
                matches = [exact]

        if not matches:
            logger.write(f"[WARN] No master dataset CSV found for platform: {platform}")
            logger.write(f"[WARN] Checked folder: {analysis_dir.resolve()}")
            continue

        if len(matches) > 1:
            logger.write(f"[WARN] Multiple master-like CSV files found for {platform}; using {matches[0].name}")
            for m in matches:
                logger.write(f"       candidate: {m.name}")

        chosen = matches[0]
        logger.write(f"[INFO] Selected master file for {platform}: {chosen.resolve()}")
        found.append((platform, chosen))

    return found


def load_and_standardize(platform: str, path: Path):
    df = pd.read_csv(path)

    rename_dict = {col: COLUMN_RENAME_MAP[col] for col in df.columns if col in COLUMN_RENAME_MAP}
    if rename_dict:
        df = df.rename(columns=rename_dict)

    if "Platform" not in df.columns:
        df["Platform"] = platform
    else:
        df["Platform"] = df["Platform"].fillna(platform).astype(str)
        df.loc[df["Platform"].str.strip().eq(""), "Platform"] = platform

    sector = SECTOR_MAP.get(platform.lower(), "Unknown")
    if "Sector" not in df.columns:
        df["Sector"] = sector
    else:
        df["Sector"] = df["Sector"].fillna(sector).astype(str)
        df.loc[df["Sector"].str.strip().eq(""), "Sector"] = sector

    df["SourceFile"] = str(path)
    return df


def validate_master(df: pd.DataFrame):
    report = []
    report.append(f"Total rows: {len(df)}")
    report.append(f"Total columns: {len(df.columns)}")

    missing_core = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing_core:
        report.append(f"[WARN] Missing core columns: {missing_core}")
    else:
        report.append("All core columns present.")

    logical_key = ["Participant", "Platform", "Task"]
    if all(col in df.columns for col in logical_key):
        dup_count = int(df.duplicated(subset=logical_key).sum())
        report.append(f"Duplicate rows on {logical_key}: {dup_count}")
    else:
        report.append(f"[WARN] Could not check duplicates on {logical_key} because some columns are missing.")

    for col in CORE_COLUMNS:
        if col in df.columns:
            report.append(f"Missing values in {col}: {int(df[col].isna().sum())}")

    if "Platform" in df.columns:
        report.append("")
        report.append("Rows by Platform:")
        platform_counts = df["Platform"].value_counts(dropna=False).sort_index()
        for platform, count in platform_counts.items():
            report.append(f"  {platform}: {count}")

    if "Platform" in df.columns and "Task" in df.columns:
        report.append("")
        report.append("Rows by Platform x Task:")
        counts = df.groupby(["Platform", "Task"]).size().sort_index()
        for (platform, task), count in counts.items():
            report.append(f"  {platform} | {task}: {count}")

    return report


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = LOGS_DIR / f"merge_master_datasets_{timestamp}.log"
    logger = TeeLogger(logfile)

    try:
        logger.write("=== Merge master datasets started ===")
        logger.write(f"Working directory: {Path.cwd()}")
        logger.write(f"Expected data directory: {DATA_DIR.resolve()}")
        logger.write(f"Results directory: {RESULTS_DIR.resolve()}")

        discovered = find_master_files(DATA_DIR, logger)

        if not discovered:
            logger.write("[ERROR] No platform master datasets were found.")
            logger.write("[ERROR] Expected pattern: data/<platform>/analysis_output/*_master_dataset.csv")
            logger.write("[ERROR] Or fallback names containing 'master'.")
            return

        report_lines = [
            "Merge report",
            "=" * 60,
            f"Run timestamp: {timestamp}",
            f"Working directory: {Path.cwd()}",
            f"Data directory: {DATA_DIR.resolve()}",
            "",
        ]

        all_frames = []

        for platform, path in discovered:
            logger.write(f"[INFO] Loading {platform}: {path}")
            try:
                df = load_and_standardize(platform, path)
                logger.write(f"[OK] {platform}: {len(df)} rows, {len(df.columns)} columns")
                report_lines.append(f"{platform}: loaded {len(df)} rows from {path}")
                all_frames.append(df)
            except Exception as e:
                msg = f"[ERROR] Failed to load {path}: {e}"
                logger.write(msg)
                report_lines.append(msg)

        if not all_frames:
            logger.write("[ERROR] No datasets were loaded successfully.")
            return

        master = pd.concat(all_frames, ignore_index=True, sort=False)

        ordered_cols = [c for c in CORE_COLUMNS if c in master.columns]
        remaining_cols = [c for c in master.columns if c not in ordered_cols]
        master = master[ordered_cols + remaining_cols]

        output_csv = MASTER_DIR / "master_dataset.csv"
        master.to_csv(output_csv, index=False)

        report_lines.append("")
        report_lines.append("Validation")
        report_lines.append("-" * 60)
        report_lines.extend(validate_master(master))

        report_path = SUMMARIES_DIR / "merge_report.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

        logger.write("")
        logger.write(f"Saved merged master dataset to: {output_csv.resolve()}")
        logger.write(f"Saved summary to: {report_path.resolve()}")
        logger.write("Finished.")

    except Exception as e:
        logger.write("[FATAL] Unhandled exception:")
        logger.write(f"Error: {e}")
        logger.write(traceback.format_exc())

    finally:
        logger.write(f"Log file: {logfile.resolve()}")
        logger.close()


if __name__ == "__main__":
    main()