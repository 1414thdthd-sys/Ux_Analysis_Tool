import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent.parent

BASE_DIR = APP_DIR / "data" / "myhealth"
OUTPUT_DIR = BASE_DIR / "analysis_output"
GRAPH_DIR = OUTPUT_DIR / "graphs"
THESIS_GRAPH_DIR = OUTPUT_DIR / "graphs_thesis"
LOG_FILE = OUTPUT_DIR / "debug_log.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
THESIS_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

PLATFORM = "myhealth"
SECTOR = "Public"
TASK_PREFIX = "H"


# =========================
# UTILITIES
# =========================
def log(msg: str):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def read_csv_safe(path: Path):
    if not path.exists():
        log(f"[WARN] Missing file: {path}")
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        log(f"[OK] Read {path.name} with {len(df)} rows")
        return df
    except Exception as e:
        log(f"[WARN] Could not read {path}: {e}")
        return None


def get_time_col(df):
    if df is None:
        return None
    for c in df.columns:
        if "Time" in c:
            return c
    return None


def parse_time(df):
    if df is None:
        return None

    df = df.copy()
    tcol = get_time_col(df)
    if tcol is None:
        return df

    dt = pd.to_datetime(df[tcol], errors="coerce", utc=True)

    try:
        dt = dt.dt.tz_convert(None)
    except Exception:
        pass

    df[tcol] = dt
    df = df.dropna(subset=[tcol]).sort_values(tcol).reset_index(drop=True)
    return df


def to_naive_timestamp(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def slice_window(df, start_ts, end_ts):
    if df is None:
        return None
    tcol = get_time_col(df)
    if tcol is None:
        return None

    start_ts = to_naive_timestamp(start_ts)
    end_ts = to_naive_timestamp(end_ts)

    out = df[(df[tcol] >= start_ts) & (df[tcol] <= end_ts)].copy()
    return out.reset_index(drop=True)


def baseline_window(df, start_ts, seconds=30):
    if df is None:
        return None
    tcol = get_time_col(df)
    if tcol is None:
        return None

    start_ts = to_naive_timestamp(start_ts)
    t0 = start_ts - pd.Timedelta(seconds=seconds)

    out = df[(df[tcol] >= t0) & (df[tcol] < start_ts)].copy()
    return out.reset_index(drop=True)


def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.mean() if not s.dropna().empty else np.nan


def safe_max(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.max() if not s.dropna().empty else np.nan


def safe_ratio(valid_count, total_count):
    if total_count == 0:
        return np.nan
    return valid_count / total_count


def normalize_series(series, method="zscore"):
    s = pd.to_numeric(series, errors="coerce").astype(float)

    if s.dropna().empty:
        return s

    if method == "zscore":
        mean = s.mean()
        std = s.std()
        if pd.isna(std) or std == 0:
            return s - mean
        return (s - mean) / std

    if method == "minmax":
        smin = s.min()
        smax = s.max()
        if pd.isna(smin) or pd.isna(smax) or smax == smin:
            return s - smin
        return (s - smin) / (smax - smin)

    return s


# =========================
# EVENTS / TASK SEGMENTS
# =========================
def find_event_col(events_df):
    candidates = [c for c in events_df.columns if "Event" in c]
    if candidates:
        return candidates[0]

    time_col = get_time_col(events_df)
    other_cols = [c for c in events_df.columns if c != time_col]
    return other_cols[0] if other_cols else None


def extract_segments(events_df, participant):
    segments = []

    if events_df is None or events_df.empty:
        return segments

    time_col = get_time_col(events_df)
    event_col = find_event_col(events_df)

    log(f"[DEBUG] {participant} events columns: {list(events_df.columns)}")
    log(f"[DEBUG] {participant} time_col={time_col}, event_col={event_col}")

    if time_col is None or event_col is None:
        return segments

    active = None

    for _, row in events_df.iterrows():
        label = str(row[event_col]).strip()
        ts = row[time_col]

        m_start = re.match(rf"^({TASK_PREFIX}\d+)_start$", label, flags=re.IGNORECASE)
        m_end = re.match(rf"^({TASK_PREFIX}\d+)_end$", label, flags=re.IGNORECASE)

        if m_start:
            active = {
                "Participant": participant,
                "Platform": PLATFORM,
                "Sector": SECTOR,
                "Task": m_start.group(1).upper(),
                "Start": ts,
                "End": None,
                "Status": None,
            }
            log(f"[DEBUG] {participant} START {active['Task']} at {ts}")
            continue

        if m_end and active is not None:
            task_name = m_end.group(1).upper()
            if task_name == active["Task"]:
                active["End"] = ts
                active["Status"] = "Success"
                segments.append(active)
                log(f"[DEBUG] {participant} END {task_name} at {ts}")
                active = None
            continue

        if label == "Incomplete_task_end" and active is not None:
            active["End"] = ts
            active["Status"] = "Failed"
            segments.append(active)
            log(f"[DEBUG] {participant} FAILED {active['Task']} at {ts}")
            active = None

    return segments


# =========================
# EYE TRACKING METRICS
# =========================
def estimate_fixation_metrics(eye_task):
    result = {
        "GazeValidCount": np.nan,
        "GazeValidRatio": np.nan,
        "FixPointValidCount": np.nan,
        "FixPointValidRatio": np.nan,
        "EstimatedFixationEpisodes": np.nan,
        "EstimatedFixationMeanDuration_ms": np.nan,
        "Movement_FIXATION_Ratio": np.nan,
        "Movement_SACCADE_Ratio": np.nan,
        "Movement_UNDEFINED_Ratio": np.nan,
    }

    if eye_task is None or eye_task.empty:
        return result

    tcol = get_time_col(eye_task)
    total = len(eye_task)

    if "GazepointX" in eye_task.columns and "GazepointY" in eye_task.columns:
        gaze_valid = eye_task["GazepointX"].notna() & eye_task["GazepointY"].notna()
        result["GazeValidCount"] = int(gaze_valid.sum())
        result["GazeValidRatio"] = safe_ratio(int(gaze_valid.sum()), total)

    if "FixationpointX" in eye_task.columns and "FixationpointY" in eye_task.columns:
        fix_valid = eye_task["FixationpointX"].notna() & eye_task["FixationpointY"].notna()
        result["FixPointValidCount"] = int(fix_valid.sum())
        result["FixPointValidRatio"] = safe_ratio(int(fix_valid.sum()), total)

    if "MovementState" not in eye_task.columns or tcol is None:
        return result

    states = eye_task["MovementState"].astype(str).str.upper()

    result["Movement_FIXATION_Ratio"] = (states == "FIXATION").mean() if len(states) else np.nan
    result["Movement_SACCADE_Ratio"] = (states == "SACCADE").mean() if len(states) else np.nan
    result["Movement_UNDEFINED_Ratio"] = (states == "UNDEFINED").mean() if len(states) else np.nan

    fix_mask = (states == "FIXATION").fillna(False)
    starts = fix_mask & ~fix_mask.shift(1, fill_value=False)
    episode_ids = starts.cumsum()
    episode_ids = episode_ids.where(fix_mask, np.nan)

    durations = []
    episode_series = pd.Series(episode_ids, index=eye_task.index)

    for ep in episode_series.dropna().unique():
        block = eye_task[episode_series == ep]
        if len(block) >= 2:
            dur_ms = (block[tcol].iloc[-1] - block[tcol].iloc[0]).total_seconds() * 1000
            durations.append(dur_ms)

    result["EstimatedFixationEpisodes"] = int(len(durations)) if durations else 0
    result["EstimatedFixationMeanDuration_ms"] = float(np.mean(durations)) if durations else np.nan

    return result


# =========================
# EXISTING PLOTS
# =========================
def save_line(df, y_col, title, outpath, normalize=True):
    if df is None or y_col not in df.columns:
        return
    tcol = get_time_col(df)
    y = pd.to_numeric(df[y_col], errors="coerce")
    if tcol is None or y.dropna().empty:
        return

    if normalize:
        y = normalize_series(y, method="zscore")

    plt.figure(figsize=(10, 4))
    plt.plot(df[tcol], y)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(f"{y_col} (z-score)" if normalize else y_col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_heatmap(eye_task, x_col, y_col, title, outpath):
    if eye_task is None or x_col not in eye_task.columns or y_col not in eye_task.columns:
        return

    x = pd.to_numeric(eye_task[x_col], errors="coerce")
    y = pd.to_numeric(eye_task[y_col], errors="coerce")
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]

    if len(x) < 5:
        return

    plt.figure(figsize=(7, 5))
    plt.hist2d(x, y, bins=50)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_multimodal(cog_df, ppg_df, eda_df, outpath, title):
    plt.figure(figsize=(12, 5))
    plotted = False

    if cog_df is not None and "Cognitive Load" in cog_df.columns:
        y = normalize_series(cog_df["Cognitive Load"], method="zscore")
        if not y.dropna().empty:
            plt.plot(
                cog_df[get_time_col(cog_df)],
                y,
                label="Cognitive Load",
            )
            plotted = True

    if ppg_df is not None and "PPG: Heart rate (bpm)" in ppg_df.columns:
        y = normalize_series(ppg_df["PPG: Heart rate (bpm)"], method="zscore")
        if not y.dropna().empty:
            plt.plot(
                ppg_df[get_time_col(ppg_df)],
                y,
                label="Heart Rate",
            )
            plotted = True

    if eda_df is not None and "EDA:Tonic" in eda_df.columns:
        y = normalize_series(eda_df["EDA:Tonic"], method="zscore")
        if not y.dropna().empty:
            plt.plot(
                eda_df[get_time_col(eda_df)],
                y,
                label="EDA Tonic",
            )
            plotted = True

    if plotted:
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Standardized Signal Value (z-score)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)

    plt.close()


def summary_bar(df, x, y, title, outname):
    if y not in df.columns:
        return

    vals = pd.to_numeric(df[y], errors="coerce")
    if vals.dropna().empty:
        return

    plt.figure(figsize=(8, 4))
    plt.bar(df[x], vals)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / outname, dpi=200)
    plt.close()


# =========================
# NEW THESIS PLOTS
# =========================
def normalize_and_interpolate(df, value_col, n_points=100):
    if df is None or df.empty or value_col not in df.columns:
        return None

    tcol = get_time_col(df)
    if tcol is None:
        return None

    y = pd.to_numeric(df[value_col], errors="coerce")
    valid = y.notna()

    if valid.sum() < 2:
        return None

    df_valid = df.loc[valid].copy()
    y_valid = y.loc[valid]

    t = df_valid[tcol]
    t_sec = (t - t.iloc[0]).dt.total_seconds()

    if len(t_sec) < 2 or t_sec.iloc[-1] <= 0:
        return None

    t_norm = t_sec / t_sec.iloc[-1]
    common_time = np.linspace(0, 1, n_points)

    try:
        interp = np.interp(common_time, t_norm, y_valid)
        return interp
    except Exception:
        return None


def aggregate_task_signal(task_dfs, value_col, n_points=100):
    curves = []

    for df in task_dfs:
        arr = normalize_and_interpolate(df, value_col, n_points=n_points)
        if arr is not None:
            curves.append(arr)

    if not curves:
        return None

    return np.nanmean(np.vstack(curves), axis=0)


def zscore(signal):
    if signal is None:
        return None
    signal = np.asarray(signal, dtype=float)
    if np.all(np.isnan(signal)):
        return None
    mean = np.nanmean(signal)
    std = np.nanstd(signal)
    if np.isnan(std) or std == 0:
        return signal - mean
    return (signal - mean) / std


def save_task_multimodal_aggregated(task, cog_dfs, eda_dfs, hr_dfs, outpath):
    x = np.linspace(0, 100, 100)

    cog_mean = aggregate_task_signal(cog_dfs, "Cognitive Load")
    eda_mean = aggregate_task_signal(eda_dfs, "EDA:Tonic")
    hr_mean = aggregate_task_signal(hr_dfs, "PPG: Heart rate (bpm)")

    cog_plot = zscore(cog_mean)
    eda_plot = zscore(eda_mean)
    hr_plot = zscore(hr_mean)

    if hr_plot is not None:
        hr_plot = pd.Series(hr_plot).rolling(5, min_periods=1).mean().values

    if cog_plot is None and eda_plot is None and hr_plot is None:
        return

    plt.figure(figsize=(10, 5))

    if cog_plot is not None:
        plt.plot(x, cog_plot, label="Cognitive Load")

    if eda_plot is not None:
        plt.plot(x, eda_plot, label="EDA")

    if hr_plot is not None:
        plt.plot(x, hr_plot, label="Heart Rate")

    plt.title(f"{task} – Aggregated Multimodal Response")
    plt.xlabel("Task Progress (%)")
    plt.ylabel("Standardized Signal Value (z-score)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def save_task_cognitive_aggregated(task, cog_dfs, outpath):
    x = np.linspace(0, 100, 100)
    cog_mean = aggregate_task_signal(cog_dfs, "Cognitive Load")

    if cog_mean is None:
        return

    cog_plot = zscore(cog_mean)

    plt.figure(figsize=(10, 4))
    plt.plot(x, cog_plot)
    plt.title(f"{task} – Cognitive Load (Aggregated)")
    plt.xlabel("Task Progress (%)")
    plt.ylabel("Cognitive Load (z-score)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def save_task_cognitive_variability(task, cog_dfs, outpath):
    x = np.linspace(0, 100, 100)
    curves = []

    for df in cog_dfs:
        arr = normalize_and_interpolate(df, "Cognitive Load", n_points=100)
        if arr is not None:
            curves.append(arr)

    if not curves:
        return

    curves = np.vstack(curves)

    z_curves = []
    for row in curves:
        mean = np.nanmean(row)
        std = np.nanstd(row)

        if np.isnan(mean):
            continue
        if np.isnan(std) or std == 0:
            z_curves.append(row - mean)
        else:
            z_curves.append((row - mean) / std)

    if not z_curves:
        return

    z_curves = np.vstack(z_curves)
    std_curve = np.nanstd(z_curves, axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(x, std_curve)
    plt.title(f"{task} – Cognitive Load Variability")
    plt.xlabel("Task Progress (%)")
    plt.ylabel("Std. Dev. of Cognitive Load (z-score)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log(f"[INFO] BASE_DIR = {BASE_DIR}")
    log(f"[INFO] OUTPUT_DIR = {OUTPUT_DIR}")
    log(f"[INFO] GRAPH_DIR = {GRAPH_DIR}")
    log(f"[INFO] THESIS_GRAPH_DIR = {THESIS_GRAPH_DIR}")

    if not BASE_DIR.exists():
        log("[ERROR] BASE_DIR does not exist")
        return

    rows = []
    task_data = {}

    participants = sorted(
        [p for p in BASE_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")],
        key=lambda x: x.name,
    )
    log(f"[INFO] Found {len(participants)} participant folders")

    for pdir in participants:
        participant = pdir.name
        log(f"\n[INFO] Processing {participant}")

        events = parse_time(read_csv_safe(pdir / f"{participant}_events.csv"))
        eye = parse_time(read_csv_safe(pdir / f"{participant}_eyetracking.csv"))
        cog = parse_time(read_csv_safe(pdir / f"{participant}_cognitiveload.csv"))
        eda = parse_time(read_csv_safe(pdir / f"{participant}_eda.csv"))
        ppg = parse_time(read_csv_safe(pdir / f"{participant}_ppg.csv"))
        face = parse_time(read_csv_safe(pdir / f"{participant}_facialexpressions.csv"))

        if events is None or events.empty:
            log(f"[WARN] Missing or empty events for {participant}")
            continue

        segments = extract_segments(events, participant)
        log(f"[INFO] {participant} segments found: {len(segments)}")

        if not segments:
            log(f"[WARN] No H-task segments found for {participant}")
            continue

        for seg in segments:
            start_ts = seg["Start"]
            end_ts = seg["End"]

            eye_task = slice_window(eye, start_ts, end_ts)
            cog_task = slice_window(cog, start_ts, end_ts)
            eda_task = slice_window(eda, start_ts, end_ts)
            ppg_task = slice_window(ppg, start_ts, end_ts)
            face_task = slice_window(face, start_ts, end_ts)
            ppg_base = baseline_window(ppg, start_ts, seconds=30)

            task = seg["Task"]
            if task not in task_data:
                task_data[task] = {"cog": [], "eda": [], "hr": []}

            if cog_task is not None and not cog_task.empty:
                task_data[task]["cog"].append(cog_task)
            if eda_task is not None and not eda_task.empty:
                task_data[task]["eda"].append(eda_task)
            if ppg_task is not None and not ppg_task.empty:
                task_data[task]["hr"].append(ppg_task)

            row = {
                "Participant": participant,
                "Platform": seg["Platform"],
                "Sector": seg["Sector"],
                "Task": task,
                "Start": start_ts,
                "End": end_ts,
                "Status": seg["Status"],
                "Success": 1 if seg["Status"] == "Success" else 0,
                "CompletionTime_sec": (end_ts - start_ts).total_seconds(),
            }

            # Cognitive
            if cog_task is not None and not cog_task.empty:
                row["CogLoadMean"] = safe_mean(cog_task["Cognitive Load"]) if "Cognitive Load" in cog_task.columns else np.nan
                row["CogLoadPeak"] = safe_max(cog_task["Cognitive Load"]) if "Cognitive Load" in cog_task.columns else np.nan
                row["ConfidenceMean"] = safe_mean(cog_task["Confidence"]) if "Confidence" in cog_task.columns else np.nan
                row["PupilDiameterMean"] = safe_mean(cog_task["Pupil Diameter"]) if "Pupil Diameter" in cog_task.columns else np.nan
                row["PupilLightReflexMean"] = safe_mean(cog_task["Pupil Light Reflex"]) if "Pupil Light Reflex" in cog_task.columns else np.nan
                row["BrightnessMean"] = safe_mean(cog_task["Brightness"]) if "Brightness" in cog_task.columns else np.nan
            else:
                row["CogLoadMean"] = np.nan
                row["CogLoadPeak"] = np.nan
                row["ConfidenceMean"] = np.nan
                row["PupilDiameterMean"] = np.nan
                row["PupilLightReflexMean"] = np.nan
                row["BrightnessMean"] = np.nan

            # Heart
            if ppg_task is not None and not ppg_task.empty:
                row["HR_Mean"] = safe_mean(ppg_task["PPG: Heart rate (bpm)"]) if "PPG: Heart rate (bpm)" in ppg_task.columns else np.nan
                row["HR_Peak"] = safe_max(ppg_task["PPG: Heart rate (bpm)"]) if "PPG: Heart rate (bpm)" in ppg_task.columns else np.nan
                row["IBI_Mean"] = safe_mean(ppg_task["PPG: Interbeat interval (ms)"]) if "PPG: Interbeat interval (ms)" in ppg_task.columns else np.nan
            else:
                row["HR_Mean"] = np.nan
                row["HR_Peak"] = np.nan
                row["IBI_Mean"] = np.nan

            if ppg_base is not None and not ppg_base.empty:
                row["HR_Baseline"] = safe_mean(ppg_base["PPG: Heart rate (bpm)"]) if "PPG: Heart rate (bpm)" in ppg_base.columns else np.nan
                row["IBI_Baseline"] = safe_mean(ppg_base["PPG: Interbeat interval (ms)"]) if "PPG: Interbeat interval (ms)" in ppg_base.columns else np.nan
            else:
                row["HR_Baseline"] = np.nan
                row["IBI_Baseline"] = np.nan

            row["HR_Delta"] = (
                row["HR_Mean"] - row["HR_Baseline"]
                if pd.notna(row["HR_Mean"]) and pd.notna(row["HR_Baseline"])
                else np.nan
            )
            row["IBI_Delta"] = (
                row["IBI_Mean"] - row["IBI_Baseline"]
                if pd.notna(row["IBI_Mean"]) and pd.notna(row["IBI_Baseline"])
                else np.nan
            )

            # EDA
            if eda_task is not None and not eda_task.empty:
                tonic = pd.to_numeric(eda_task["EDA:Tonic"], errors="coerce") if "EDA:Tonic" in eda_task.columns else pd.Series(dtype=float)
                peakr = pd.to_numeric(eda_task["EDA:Peak rate"], errors="coerce") if "EDA:Peak rate" in eda_task.columns else pd.Series(dtype=float)

                row["EDA_TonicMean"] = tonic.mean() if not tonic.dropna().empty else np.nan
                row["EDA_PeakRateMean"] = peakr.mean() if not peakr.dropna().empty else np.nan
                row["EDA_PeakRateMax"] = peakr.max() if not peakr.dropna().empty else np.nan
                row["EDA_ValidRatio"] = tonic.notna().mean() if len(tonic) else np.nan
            else:
                row["EDA_TonicMean"] = np.nan
                row["EDA_PeakRateMean"] = np.nan
                row["EDA_PeakRateMax"] = np.nan
                row["EDA_ValidRatio"] = np.nan

            # Facial
            face_cols = [
                "Facial expression: Neutral",
                "Facial expression: Happy",
                "Facial expression: Sad",
                "Facial expression: Angry",
                "Facial expression: Surprised",
                "Facial expression: Scared",
                "Facial expression: Disgusted",
                "Facial expression: Confused",
                "Facial expression: Bored",
                "Facial expression: Interested",
            ]

            if face_task is not None and not face_task.empty:
                for c in face_cols:
                    out_name = c.replace("Facial expression: ", "").replace(" ", "") + "Mean"
                    row[out_name] = safe_mean(face_task[c]) if c in face_task.columns else np.nan

                neg_vals = []
                for c in [
                    "Facial expression: Sad",
                    "Facial expression: Angry",
                    "Facial expression: Disgusted",
                ]:
                    if c in face_task.columns:
                        val = safe_mean(face_task[c])
                        if pd.notna(val):
                            neg_vals.append(val)

                row["NegativeAffectMean"] = np.mean(neg_vals) if neg_vals else np.nan
            else:
                for c in face_cols:
                    out_name = c.replace("Facial expression: ", "").replace(" ", "") + "Mean"
                    row[out_name] = np.nan
                row["NegativeAffectMean"] = np.nan

            # Eye tracking
            row.update(estimate_fixation_metrics(eye_task))
            rows.append(row)

            # Existing per participant x task graphs
            prefix = f"{participant}_{task}"
            save_line(cog_task, "Cognitive Load", f"{prefix} Cognitive Load", GRAPH_DIR / f"{prefix}_cognitive_load.png", normalize=True)
            save_line(cog_task, "Pupil Diameter", f"{prefix} Pupil Diameter", GRAPH_DIR / f"{prefix}_pupil.png", normalize=True)
            save_line(ppg_task, "PPG: Heart rate (bpm)", f"{prefix} Heart Rate", GRAPH_DIR / f"{prefix}_hr.png", normalize=True)
            save_line(eda_task, "EDA:Tonic", f"{prefix} EDA Tonic", GRAPH_DIR / f"{prefix}_eda.png", normalize=True)
            save_line(face_task, "Facial expression: Confused", f"{prefix} Confusion", GRAPH_DIR / f"{prefix}_confusion.png", normalize=True)
            save_heatmap(eye_task, "GazepointX", "GazepointY", f"{prefix} Gaze Heatmap", GRAPH_DIR / f"{prefix}_gaze_heatmap.png")
            save_heatmap(eye_task, "FixationpointX", "FixationpointY", f"{prefix} Fixation Heatmap", GRAPH_DIR / f"{prefix}_fixation_heatmap.png")
            save_multimodal(cog_task, ppg_task, eda_task, GRAPH_DIR / f"{prefix}_multimodal.png", f"{prefix} Multimodal Timeline")

    # Save master dataset
    master_df = pd.DataFrame(rows)
    master_path = OUTPUT_DIR / "myhealth_master_dataset.csv"
    summary_path = OUTPUT_DIR / "myhealth_task_summary.csv"

    log(f"\n[INFO] Total task rows collected: {len(master_df)}")
    master_df.to_csv(master_path, index=False)
    log(f"[INFO] Saved master dataset to: {master_path}")

    # Save summary
    if not master_df.empty:
        summary = master_df.groupby("Task", dropna=False).agg(
            # Counts
            N=("Participant", "count"),

            # Performance
            SuccessRate=("Success", "mean"),
            CompletionTimeMean_sec=("CompletionTime_sec", "mean"),
            CompletionTimeMedian_sec=("CompletionTime_sec", "median"),
            CompletionTimeStd_sec=("CompletionTime_sec", "std"),

            # Cognitive
            CogLoadMean=("CogLoadMean", "mean"),
            CogLoadMedian=("CogLoadMean", "median"),
            CogLoadStd=("CogLoadMean", "std"),

            CogLoadPeakMean=("CogLoadPeak", "mean"),
            CogLoadPeakMedian=("CogLoadPeak", "median"),
            CogLoadPeakStd=("CogLoadPeak", "std"),

            # Heart rate
            HR_MeanMean=("HR_Mean", "mean"),
            HR_MeanMedian=("HR_Mean", "median"),
            HR_MeanStd=("HR_Mean", "std"),

            HR_DeltaMean=("HR_Delta", "mean"),
            HR_DeltaMedian=("HR_Delta", "median"),
            HR_DeltaStd=("HR_Delta", "std"),

            # EDA
            EDA_TonicMean=("EDA_TonicMean", "mean"),
            EDA_TonicMedian=("EDA_TonicMean", "median"),
            EDA_TonicStd=("EDA_TonicMean", "std"),

            EDA_PeakRateMean=("EDA_PeakRateMean", "mean"),
            EDA_PeakRateMedian=("EDA_PeakRateMean", "median"),
            EDA_PeakRateStd=("EDA_PeakRateMean", "std"),

            # Facial
            ConfusedMean=("ConfusedMean", "mean"),
            ConfusedMedian=("ConfusedMean", "median"),
            ConfusedStd=("ConfusedMean", "std"),

            # Eye tracking
            EstimatedFixationEpisodesMean=("EstimatedFixationEpisodes", "mean"),
            EstimatedFixationEpisodesMedian=("EstimatedFixationEpisodes", "median"),
            EstimatedFixationEpisodesStd=("EstimatedFixationEpisodes", "std"),

            EstimatedFixationMeanDuration_ms_Mean=("EstimatedFixationMeanDuration_ms", "mean"),
            EstimatedFixationMeanDuration_ms_Median=("EstimatedFixationMeanDuration_ms", "median"),
            EstimatedFixationMeanDuration_ms_Std=("EstimatedFixationMeanDuration_ms", "std"),

            GazeValidRatioMean=("GazeValidRatio", "mean"),
            GazeValidRatioMedian=("GazeValidRatio", "median"),
            GazeValidRatioStd=("GazeValidRatio", "std"),
        ).reset_index()

        summary.to_csv(summary_path, index=False)
        log(f"[INFO] Saved task summary to: {summary_path}")

        summary_bar(summary, "Task", "CompletionTimeMean_sec", "Health Mean Completion Time", "summary_completion_time.png")
        summary_bar(summary, "Task", "SuccessRate", "Health Success Rate", "summary_success_rate.png")
        summary_bar(summary, "Task", "CogLoadMean", "Health Mean Cognitive Load", "summary_cognitive_load.png")
        summary_bar(summary, "Task", "CogLoadPeakMean", "Health Mean Cognitive Peak", "summary_cognitive_peak.png")
        summary_bar(summary, "Task", "HR_DeltaMean", "Health Mean HR Delta", "summary_hr_delta.png")
        summary_bar(summary, "Task", "HR_MeanMean", "Health Mean HR", "summary_hr_mean.png")
        summary_bar(summary, "Task", "EDA_TonicMean", "Health Mean EDA Tonic", "summary_eda_tonic.png")
        summary_bar(summary, "Task", "EDA_PeakRateMean", "Health Mean EDA Peak Rate", "summary_eda_peakrate.png")
        summary_bar(summary, "Task", "ConfusedMean", "Health Mean Confusion", "summary_confusion.png")
        summary_bar(summary, "Task", "EstimatedFixationEpisodesMean", "Health Estimated Fixation Episodes", "summary_fixation_episodes.png")
        summary_bar(summary, "Task", "EstimatedFixationMeanDuration_ms_Mean", "Health Mean Fixation Duration", "summary_fixation_duration.png")
        summary_bar(summary, "Task", "CompletionTimeStd_sec", "Health Time Variability", "summary_time_std.png")
    else:
        log("[WARN] master_df is empty, summary not created")

    # New thesis graphs
    if task_data:
        for task, data in sorted(task_data.items()):
            save_task_multimodal_aggregated(
                task=task,
                cog_dfs=data["cog"],
                eda_dfs=data["eda"],
                hr_dfs=data["hr"],
                outpath=THESIS_GRAPH_DIR / f"{task}_multimodal_aggregated.png",
            )

            save_task_cognitive_aggregated(
                task=task,
                cog_dfs=data["cog"],
                outpath=THESIS_GRAPH_DIR / f"{task}_cognitive_aggregated.png",
            )

            save_task_cognitive_variability(
                task=task,
                cog_dfs=data["cog"],
                outpath=THESIS_GRAPH_DIR / f"{task}_cognitive_variability.png",
            )

        log(f"[INFO] Thesis aggregated graphs created in: {THESIS_GRAPH_DIR}")
    else:
        log("[WARN] No task data available for thesis graphs")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback_text = traceback.format_exc()
        print(traceback_text)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(traceback_text + "\n")