import sys
import re
import subprocess
import zipfile
import shutil
from pathlib import Path
import pandas as pd


# =========================
# BASE PATH
# =========================
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent.parent

DATA_ROOT = APP_DIR / "data"
RAR_EXE = Path(r"C:\Program Files\WinRAR\Rar.exe")


# =========================
# HELPERS
# =========================
def normalize_p_folder_name(name: str):
    """
    Examples:
    P3_xxx -> P03
    P12_xxx -> P12
    P05 -> P05
    P20_-_GIOTA -> P20
    P20_-_GIOTA_bnmmx -> P20
    """
    m = re.match(r"^[Pp](\d{1,2})(?:\b|[_\- ])", name)
    if not m:
        m = re.match(r"^[Pp](\d{1,2})$", name)
    if not m:
        return None

    return f"P{int(m.group(1)):02d}"


def has_comment_rows(file_path: Path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.readline().startswith("#")
    except Exception:
        return False


def is_already_clean(file_path: Path, prefix: str):
    return file_path.stem.startswith(prefix + "_") and not has_comment_rows(file_path)


def build_final_name(file_path: Path, prefix: str):
    base_name = file_path.stem
    m = re.match(r"^[Pp]\d{2}_(.+)$", base_name)
    if m:
        base_name = m.group(1)
    return f"{prefix}_{base_name}.csv"


# =========================
# ARCHIVE EXTRACTION
# =========================
def get_new_dirs(before_set, parent_folder: Path):
    after_dirs = {p.resolve() for p in parent_folder.iterdir() if p.is_dir()}
    return [Path(p) for p in sorted(after_dirs - before_set)]


def find_single_extracted_root(folder: Path, expected_prefix: str):
    """
    Prefer a single extracted folder whose name starts with the archive stem.
    Fallback: if exactly one new folder exists, use it.
    """
    candidates = [p for p in folder.iterdir() if p.is_dir() and p.name.startswith(expected_prefix)]
    if len(candidates) == 1:
        return candidates[0]

    all_dirs = [p for p in folder.iterdir() if p.is_dir()]
    if len(all_dirs) == 1:
        return all_dirs[0]

    return None


def merge_folder_contents(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        target = dst / item.name

        if item.is_dir():
            if target.exists() and target.is_dir():
                merge_folder_contents(item, target)
                try:
                    item.rmdir()
                except Exception:
                    pass
            else:
                shutil.move(str(item), str(target))
        else:
            if target.exists():
                target.unlink()
            shutil.move(str(item), str(target))


def extract_zip_and_normalize(zip_file: Path, platform_folder: Path, normalized: str):
    before_dirs = {p.resolve() for p in platform_folder.iterdir() if p.is_dir()}

    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(platform_folder)

    new_dirs = get_new_dirs(before_dirs, platform_folder)

    if len(new_dirs) == 1:
        extracted_root = new_dirs[0]
    else:
        extracted_root = find_single_extracted_root(platform_folder, zip_file.stem)

    if extracted_root is None:
        raise RuntimeError(f"Could not identify extracted folder for {zip_file.name}")

    final_folder = platform_folder / normalized

    if extracted_root.resolve() == final_folder.resolve():
        return final_folder

    if final_folder.exists():
        merge_folder_contents(extracted_root, final_folder)
        try:
            extracted_root.rmdir()
        except Exception:
            pass
    else:
        extracted_root.rename(final_folder)

    return final_folder


def extract_rar_and_normalize(rar_file: Path, platform_folder: Path, normalized: str):
    if not RAR_EXE.exists():
        raise FileNotFoundError(f"WinRAR not found: {RAR_EXE}")

    before_dirs = {p.resolve() for p in platform_folder.iterdir() if p.is_dir()}

    result = subprocess.run(
        [
            str(RAR_EXE),
            "x",
            "-y",
            str(rar_file),
            str(platform_folder) + "\\"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or f"RAR extraction failed: {rar_file.name}")

    new_dirs = get_new_dirs(before_dirs, platform_folder)

    if len(new_dirs) == 1:
        extracted_root = new_dirs[0]
    else:
        extracted_root = find_single_extracted_root(platform_folder, rar_file.stem)

    if extracted_root is None:
        raise RuntimeError(f"Could not identify extracted folder for {rar_file.name}")

    final_folder = platform_folder / normalized

    if extracted_root.resolve() == final_folder.resolve():
        return final_folder

    if final_folder.exists():
        merge_folder_contents(extracted_root, final_folder)
        try:
            extracted_root.rmdir()
        except Exception:
            pass
    else:
        extracted_root.rename(final_folder)

    return final_folder


def extract_archives_in_platform_folder(platform_folder: Path):
    print(f"\nScanning for archives in: {platform_folder}")

    archive_files = [
        p for p in platform_folder.iterdir()
        if p.is_file() and p.suffix.lower() in [".zip", ".rar"]
    ]

    if not archive_files:
        print("No archive files found.")
        return

    print(f"Found {len(archive_files)} archive file(s):")
    for af in archive_files:
        print(f" - {af.name}")

    for archive_file in archive_files:
        normalized = normalize_p_folder_name(archive_file.stem)

        if normalized is None:
            print(f"[ERROR] Cannot determine participant folder from archive name: {archive_file.name}")
            continue

        try:
            print(f"\nExtracting: {archive_file.name}")

            if archive_file.suffix.lower() == ".zip":
                final_folder = extract_zip_and_normalize(archive_file, platform_folder, normalized)
            else:
                final_folder = extract_rar_and_normalize(archive_file, platform_folder, normalized)

            print(f"Final participant folder: {final_folder.name}")

            archive_file.unlink()
            print(f"Deleted archive: {archive_file.name}")

        except Exception as e:
            print(f"[ERROR] Exception extracting {archive_file.name}: {e}")


# =========================
# CLEAN CSV FILE
# =========================
def clean_raw_file(file_path: Path, prefix: str):
    try:
        df = pd.read_csv(
            file_path,
            sep=";",
            comment="#",
            engine="python",
            encoding="utf-8",
        )

        if df.empty:
            print(f"Skipped (empty): {file_path.name}")
            return False

        final_name = build_final_name(file_path, prefix)
        temp_file = file_path.with_name("__temp__" + final_name)
        final_file = file_path.with_name(final_name)

        df.to_csv(temp_file, index=False, encoding="utf-8-sig")

        file_path.unlink()

        if final_file.exists():
            final_file.unlink()

        temp_file.rename(final_file)

        print(f"Cleaned: {file_path.name} -> {final_name}")
        return True

    except Exception as e:
        print(f"Error with {file_path.name}: {e}")
        return False


# =========================
# PROCESS PLATFORM
# =========================
def process_project_folder(project_folder: Path):
    if not project_folder.exists() or not project_folder.is_dir():
        print(f"[ERROR] Folder not found: {project_folder}")
        return 0

    print(f"\n=== Processing project: {project_folder.name} ===")

    # Step 0: extract archives first
    extract_archives_in_platform_folder(project_folder)
    print("Archive extraction step completed.")

    # Step 1: normalize participant folders
    subfolders = [p for p in project_folder.iterdir() if p.is_dir()]

    for subfolder in subfolders:
        normalized = normalize_p_folder_name(subfolder.name)

        if normalized is None:
            print(f"Skipped folder: {subfolder.name}")
            continue

        if subfolder.name != normalized:
            target = subfolder.with_name(normalized)

            if target.exists():
                print(f"Cannot rename {subfolder.name} -> {normalized} (exists)")
            else:
                old_name = subfolder.name
                subfolder.rename(target)
                print(f"Renamed: {old_name} -> {normalized}")

    # Refresh after renaming
    subfolders = [p for p in project_folder.iterdir() if p.is_dir()]

    cleaned_count = 0

    # Step 2: clean files
    for subfolder in subfolders:
        prefix = normalize_p_folder_name(subfolder.name)

        if prefix is None:
            continue

        csv_files = list(subfolder.glob("*.csv"))

        if not csv_files:
            print(f"No CSV files in: {project_folder.name}/{subfolder.name}")
            continue

        print(f"\nChecking: {project_folder.name}/{subfolder.name}")

        for file_path in csv_files:
            if is_already_clean(file_path, prefix):
                print(f"OK (already clean): {file_path.name}")
                continue

            ok = clean_raw_file(file_path, prefix)
            if ok:
                cleaned_count += 1

    return cleaned_count


def process_one_platform(platform_name: str):
    return process_project_folder(DATA_ROOT / platform_name)


def process_all_platforms():
    cleaned_total = 0

    for project_folder in sorted(DATA_ROOT.iterdir()):
        if project_folder.is_dir():
            cleaned_total += process_project_folder(project_folder)

    return cleaned_total


# =========================
# MAIN
# =========================
def main():
    print("\n=== CSV CLEANER ===")
    print(f"Data root: {DATA_ROOT}\n")

    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()

        if arg.lower() == "--all":
            total = process_all_platforms()
            print(f"\nFinished. Cleaned files: {total}")
            return

        total = process_one_platform(arg)
        print(f"\nFinished. Cleaned files: {total}")
        return

    print("1 = One platform")
    print("2 = All platforms\n")

    choice = input("Choose (1/2): ").strip()

    if choice == "2":
        total = process_all_platforms()
    else:
        platform = input("Enter platform (aade, govgr, efka...): ").strip()
        total = process_one_platform(platform)

    print(f"\nFinished. Cleaned files: {total}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print(traceback.format_exc())
        input("\nPress Enter to exit...")