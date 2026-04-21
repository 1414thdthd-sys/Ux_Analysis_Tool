import os
import re
import sys
import time
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
from pathlib import Path

APP_TITLE = "UX Analysis Tool"
OUTPUT_FOLDER_NAME = "analysis_output"

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

SCRIPTS_DIR = BASE_DIR / "spython"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

SCRIPTS = {
    "Clean CSV Files": "clean_csv.py",
    "AADE": "aade.py",
    "EFKA": "efka.py",
    "GOV.GR": "govgr.py",
    "MYHEALTH": "myhealth.py",
    "TELECOMS": "telecoms.py",
    "WEBBANKING": "webbanking.py",
    "Merge Master Dataset": "merge_master_datasets.py",
    "Build Task Summary": "build_task_summary_and_rankings.py",
    "Build Platform / Sector Summary": "build_platform_sector_summaries.py",
}

PLATFORM_INFO = {
    "AADE": {"script": "aade.py", "folder": "aade"},
    "EFKA": {"script": "efka.py", "folder": "efka"},
    "GOV.GR": {"script": "govgr.py", "folder": "govgr"},
    "MYHEALTH": {"script": "myhealth.py", "folder": "myhealth"},
    "TELECOMS": {"script": "telecoms.py", "folder": "telecoms"},
    "WEBBANKING": {"script": "webbanking.py", "folder": "webbanking"},
}

ALL_PLATFORM_KEYS = list(PLATFORM_INFO.keys())
PARTICIPANT_RE = re.compile(r"\bP\d{2}\b", re.IGNORECASE)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1280x920")
        self.root.minsize(1080, 780)

        self.colors = {
            "bg": "#edf3f8",
            "panel": "#ffffff",
            "header": "#18324a",
            "header_text": "#ffffff",
            "header_sub": "#d5e1ee",
            "accent": "#2f80ed",
            "accent_dark": "#1f5fb8",
            "accent_soft": "#eaf2ff",
            "text": "#1f2937",
            "muted": "#667085",
            "border": "#cfd8e3",
            "success": "#157347",
            "success_bg": "#eaf7ee",
            "warning": "#9a6700",
            "warning_bg": "#fff4db",
            "danger": "#b42318",
            "danger_bg": "#fdecec",
            "footer": "#e2eaf2",
            "toolbar": "#f7fafc",
        }

        self._configure_style()

        self.output_queue = queue.Queue()
        self.running = False
        self.buttons = []
        self.stop_requested = False

        self.current_process = None
        self.current_task_label = ""
        self.current_platform_folder = None
        self.current_log_file = None
        self.current_log_pos = 0
        self.seen_participants = set()
        self.total_participants = 0
        self.current_participant = "-"
        self.total_tasks_in_run = 0
        self.current_task_index = 0
        self.current_task_start_time = None

        self.run_tasks = []
        self.task_status = {}
        self.task_elapsed = {}
        self.task_data_status = {}

        self._build_ui()
        self.root.after(100, self._poll_output)
        self.root.after(350, self._poll_runtime_state)

    def _configure_style(self):
        self.root.configure(bg=self.colors["bg"])
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor="#dbe5f0",
            background=self.colors["accent"],
            bordercolor="#dbe5f0",
            lightcolor=self.colors["accent"],
            darkcolor=self.colors["accent"],
        )

        style.configure(
            "Treeview",
            background="#ffffff",
            foreground=self.colors["text"],
            rowheight=28,
            fieldbackground="#ffffff",
            bordercolor=self.colors["border"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background="#e7eef7",
            foreground=self.colors["header"],
            relief="flat",
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Treeview.Heading", background=[("active", "#dce7f5")])

    # =========================
    # UI
    # =========================
    def _build_ui(self):
        header = tk.Frame(self.root, bg=self.colors["header"], height=76)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        header_inner = tk.Frame(header, bg=self.colors["header"])
        header_inner.pack(fill="both", expand=True, padx=16, pady=10)

        title = tk.Label(
            header_inner,
            text="UX Analysis Tool",
            font=("Segoe UI", 18, "bold"),
            bg=self.colors["header"],
            fg=self.colors["header_text"]
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            header_inner,
            text="Data cleaning, platform processing, merge, and summary workflow",
            font=("Segoe UI", 9),
            bg=self.colors["header"],
            fg=self.colors["header_sub"]
        )
        subtitle.pack(anchor="w", pady=(2, 0))

        footer = tk.Frame(self.root, bg=self.colors["footer"], height=40)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        tk.Label(
            footer,
            text="ICSD - University of the Aegean, 2026",
            bg=self.colors["footer"],
            fg="#334155",
            font=("Segoe UI", 9, "bold")
        ).pack(side="left", padx=10, pady=10)

        tk.Label(
            footer,
            text="Created by Th. Makris",
            bg=self.colors["footer"],
            fg="#475467",
            font=("Segoe UI", 9)
        ).pack(side="right", padx=10, pady=10)

        top = tk.Frame(self.root, bg=self.colors["bg"])
        top.pack(fill="both", expand=True, padx=12, pady=(10, 8))

        # LEFT SIDE
        left_container = tk.Frame(top, width=320, bg=self.colors["bg"])
        left_container.pack(side="left", fill="y")
        left_container.pack_propagate(False)

        left_canvas = tk.Canvas(left_container, highlightthickness=0, bg=self.colors["bg"])
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_scrollbar.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="both", expand=True)

        left = tk.Frame(left_canvas, bg=self.colors["bg"])
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_left_configure(event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def _on_canvas_configure(event):
            left_canvas.itemconfig(left_window, width=event.width)

        left.bind("<Configure>", _on_left_configure)
        left_canvas.bind("<Configure>", _on_canvas_configure)

        def _bind_mousewheel(_event):
            left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(_event):
            left_canvas.unbind_all("<MouseWheel>")

        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        left_canvas.bind("<Enter>", _bind_mousewheel)
        left_canvas.bind("<Leave>", _unbind_mousewheel)
        left.bind("<Enter>", _bind_mousewheel)
        left.bind("<Leave>", _unbind_mousewheel)

        # RIGHT SIDE
        right = tk.Frame(top, bg=self.colors["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(14, 0))

        util_frame = self._make_section(left, "Utilities")
        util_frame.pack(fill="x", pady=(0, 10))
        self._make_button(util_frame, "Clean CSV Files", self.run_clean_csv).pack(pady=4)
        self._make_button(util_frame, "Clean ALL CSV Files", self.run_clean_csv_all).pack(pady=4)

        platform_frame = self._make_section(left, "Platform Analysis")
        platform_frame.pack(fill="x", pady=(0, 10))

        for key in ALL_PLATFORM_KEYS:
            self._make_button(
                platform_frame,
                f"Run {key}",
                lambda k=key: self.run_named_script(k)
            ).pack(pady=3)

        self._make_button(
            platform_frame,
            "Run All Platform Scripts",
            self.run_all_platforms
        ).pack(pady=(8, 3))

        agg_frame = self._make_section(left, "Aggregation")
        agg_frame.pack(fill="x", pady=(0, 10))

        self._make_button(
            agg_frame,
            "Merge Master Dataset",
            lambda: self.run_named_script("Merge Master Dataset")
        ).pack(pady=4)

        self._make_button(
            agg_frame,
            "Build Task Summary",
            lambda: self.run_named_script("Build Task Summary")
        ).pack(pady=4)

        self._make_button(
            agg_frame,
            "Build Platform / Sector Summary",
            lambda: self.run_named_script("Build Platform / Sector Summary")
        ).pack(pady=4)

        self._make_button(
            agg_frame,
            "Run Merge + Summary",
            self.run_merge_and_summary
        ).pack(pady=(8, 4))

        self._make_button(
            agg_frame,
            "Run FULL PIPELINE",
            self.run_full_pipeline
        ).pack(pady=(8, 4))

        tk.Frame(left, height=10, bg=self.colors["bg"]).pack(fill="x")

        status_frame = self._make_section(right, "Status")
        status_frame.pack(fill="x", pady=(0, 8))

        self.status_var = tk.StringVar(value="Ready")
        self.task_var = tk.StringVar(value="Current task: -")
        self.participant_var = tk.StringVar(value="Participant: -")
        self.elapsed_var = tk.StringVar(value="Elapsed: 00:00:00")
        self.progress_text_var = tk.StringVar(value="Task progress: 0 / 0")
        self.participant_progress_text_var = tk.StringVar(value="Participant progress: 0 / 0")

        self.status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 11, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["accent_dark"]
        )
        self.status_label.pack(anchor="w")

        tk.Label(status_frame, textvariable=self.task_var, font=("Segoe UI", 10), bg=self.colors["panel"], fg=self.colors["text"]).pack(anchor="w", pady=(4, 0))
        tk.Label(status_frame, textvariable=self.participant_var, font=("Segoe UI", 10), bg=self.colors["panel"], fg=self.colors["text"]).pack(anchor="w", pady=(2, 0))
        tk.Label(status_frame, textvariable=self.elapsed_var, font=("Segoe UI", 10), bg=self.colors["panel"], fg=self.colors["text"]).pack(anchor="w", pady=(2, 6))

        tk.Label(status_frame, textvariable=self.progress_text_var, bg=self.colors["panel"], fg=self.colors["muted"]).pack(anchor="w")
        self.task_progress = ttk.Progressbar(status_frame, mode="determinate", maximum=100, style="Custom.Horizontal.TProgressbar")
        self.task_progress.pack(fill="x", pady=(2, 8))

        tk.Label(status_frame, textvariable=self.participant_progress_text_var, bg=self.colors["panel"], fg=self.colors["muted"]).pack(anchor="w")
        self.participant_progress = ttk.Progressbar(status_frame, mode="determinate", maximum=100, style="Custom.Horizontal.TProgressbar")
        self.participant_progress.pack(fill="x", pady=(2, 2))

        status_grid_frame = self._make_section(right, "Per-Task Status")
        status_grid_frame.pack(fill="x", pady=(0, 8))

        task_tree_container = tk.Frame(status_grid_frame, bg=self.colors["panel"])
        task_tree_container.pack(fill="x", expand=False)

        task_tree_scroll = ttk.Scrollbar(task_tree_container, orient="vertical")
        task_tree_scroll.pack(side="right", fill="y")

        self.task_tree = ttk.Treeview(
            task_tree_container,
            columns=("task", "status", "elapsed", "data"),
            show="headings",
            height=3,
            yscrollcommand=task_tree_scroll.set
        )
        task_tree_scroll.config(command=self.task_tree.yview)

        self.task_tree.heading("task", text="Task")
        self.task_tree.column("task", width=230, anchor="w")
        self.task_tree.heading("status", text="Status")
        self.task_tree.column("status", width=110, anchor="center")
        self.task_tree.heading("elapsed", text="Elapsed")
        self.task_tree.column("elapsed", width=110, anchor="center")
        self.task_tree.heading("data", text="Data")
        self.task_tree.column("data", width=180, anchor="center")
        self.task_tree.pack(side="left", fill="x", expand=True)

        self.task_tree.tag_configure("pending", background="#ffffff", foreground=self.colors["muted"])
        self.task_tree.tag_configure("running", background=self.colors["accent_soft"], foreground="#175cd3")
        self.task_tree.tag_configure("done", background=self.colors["success_bg"], foreground=self.colors["success"])
        self.task_tree.tag_configure("failed", background=self.colors["danger_bg"], foreground=self.colors["danger"])
        self.task_tree.tag_configure("error", background=self.colors["danger_bg"], foreground=self.colors["danger"])
        self.task_tree.tag_configure("stopped", background=self.colors["warning_bg"], foreground=self.colors["warning"])
        self.task_tree.tag_configure("stopping", background=self.colors["warning_bg"], foreground=self.colors["warning"])
        self.task_tree.tag_configure("precheck failed", background=self.colors["warning_bg"], foreground=self.colors["warning"])

        log_frame = self._make_section(right, "Run Log")
        log_frame.pack(fill="both", expand=True)

        toolbar = tk.Frame(log_frame, bg=self.colors["panel"])
        toolbar.pack(fill="x", pady=(0, 6))

        tk.Button(
            toolbar, text="STOP", bg="#c62828", fg="white",
            activebackground="#a61f1f", activeforeground="white",
            relief="flat", command=self.stop_processing, padx=10, pady=6
        ).pack(side="right", padx=(0, 6))

        tk.Button(
            toolbar, text="Clear Log", command=self.clear_log,
            bg=self.colors["toolbar"], fg=self.colors["text"],
            relief="solid", bd=1, padx=10, pady=6
        ).pack(side="right", padx=(0, 6))

        tk.Button(
            toolbar, text="Open Output Folder", command=self.open_platform_output,
            bg=self.colors["toolbar"], fg=self.colors["text"],
            relief="solid", bd=1, padx=10, pady=6
        ).pack(side="right", padx=(0, 6))

        tk.Button(
            toolbar, text="Open Tool Folder", command=self.open_base_dir,
            bg=self.colors["toolbar"], fg=self.colors["text"],
            relief="solid", bd=1, padx=10, pady=6
        ).pack(side="right", padx=(0, 6))

        tk.Button(
            toolbar, text="Open Results Folder", command=self.open_results,
            bg=self.colors["toolbar"], fg=self.colors["text"],
            relief="solid", bd=1, padx=10, pady=6
        ).pack(side="right", padx=(0, 6))

        self.log_box = scrolledtext.ScrolledText(
            log_frame,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
            bg="#fbfdff",
            fg="#111827",
            insertbackground="#111827",
            relief="solid",
            bd=1
        )
        self.log_box.pack(fill="both", expand=True)

        self._log("Application started.")
        self._log(f"Base folder: {BASE_DIR}")
        self._log(f"Scripts folder: {SCRIPTS_DIR}")
        self._log(f"Data folder: {DATA_DIR}")
        self._log(f"Results folder: {RESULTS_DIR}")

    # =========================
    # Helpers
    # =========================
    def _make_section(self, parent, title):
        frame = tk.LabelFrame(
            parent,
            text=title,
            padx=8,
            pady=8,
            bg=self.colors["panel"],
            fg=self.colors["header"],
            font=("Segoe UI", 10, "bold"),
            bd=1,
            relief="solid"
        )
        return frame

    def _make_button(self, parent, text, command):
        btn = tk.Button(
            parent,
            text=text,
            width=30,
            height=2,
            command=command,
            bg=self.colors["accent"],
            fg="white",
            activebackground=self.colors["accent_dark"],
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            cursor="hand2"
        )
        self.buttons.append(btn)
        return btn

    def _set_buttons_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for btn in self.buttons:
            btn.configure(state=state)

    def _log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _format_elapsed(self, seconds):
        seconds = max(0, int(seconds))
        return time.strftime("%H:%M:%S", time.gmtime(seconds))

    def _platform_output_dir(self, folder_name):
        return DATA_DIR / folder_name / OUTPUT_FOLDER_NAME

    def _platform_master_file(self, folder_name):
        return self._platform_output_dir(folder_name) / f"{folder_name}_master_dataset.csv"

    def _platform_summary_file(self, folder_name):
        return self._platform_output_dir(folder_name) / f"{folder_name}_task_summary.csv"

    def _merge_master_file(self):
        return RESULTS_DIR / "master" / "master_dataset.csv"

    def _summary_output_file(self):
        return RESULTS_DIR / "summaries" / "task_summary.csv"

    def _platform_sector_summary_file(self):
        return RESULTS_DIR / "summaries" / "platform_summary.csv"

    def _sector_summary_file(self):
        return RESULTS_DIR / "summaries" / "sector_summary.csv"

    def _row_tag_for_status(self, status):
        if not status:
            return "pending"
        return status.lower()

    def _update_tree_row(self, task_name, status=None, elapsed=None, data_status=None):
        values = (
            task_name,
            status if status is not None else self.task_status.get(task_name, ""),
            elapsed if elapsed is not None else self.task_elapsed.get(task_name, ""),
            data_status if data_status is not None else self.task_data_status.get(task_name, ""),
        )

        row_tag = self._row_tag_for_status(values[1])

        if self.task_tree.exists(task_name):
            self.task_tree.item(task_name, values=values, tags=(row_tag,))
        else:
            self.task_tree.insert("", "end", iid=task_name, values=values, tags=(row_tag,))

    def _reset_task_tree(self, tasks):
        for item in self.task_tree.get_children():
            self.task_tree.delete(item)

        self.run_tasks = [label for label, _ in tasks]
        self.task_status = {}
        self.task_elapsed = {}
        self.task_data_status = {}

        for label, _ in tasks:
            self.task_status[label] = "Pending"
            self.task_elapsed[label] = "-"
            self.task_data_status[label] = "-"
            self._update_tree_row(label)

    def _check_script_exists(self, script_filename):
        path = SCRIPTS_DIR / script_filename
        if not path.exists():
            messagebox.showerror("Missing Script", f"Script not found:\n{path}")
            return None
        return path

    def _participant_dirs_for_folder(self, folder_name):
        platform_dir = DATA_DIR / folder_name
        if not platform_dir.exists():
            return []
        return sorted(
            [p for p in platform_dir.iterdir() if p.is_dir() and PARTICIPANT_RE.fullmatch(p.name.upper())]
        )

    def _set_running(self, value):
        self.running = value
        self._set_buttons_enabled(not value)

        if not value:
            self.status_var.set("Ready")
            self.status_label.configure(fg=self.colors["success"])
            self.task_var.set("Current task: -")
            self.participant_var.set("Participant: -")
            self.elapsed_var.set("Elapsed: 00:00:00")
            self.current_platform_folder = None
            self.current_log_file = None
            self.current_log_pos = 0
            self.current_participant = "-"
            self.seen_participants = set()
            self.total_participants = 0
            self.current_process = None
            self.current_task_start_time = None
            self.stop_requested = False

    def _update_status_color(self, text):
        lower = text.lower()
        if "running" in lower:
            self.status_label.configure(fg="#175cd3")
        elif "completed" in lower or "ready" in lower:
            self.status_label.configure(fg=self.colors["success"])
        elif "stopping" in lower or "stopped" in lower:
            self.status_label.configure(fg=self.colors["warning"])
        elif "error" in lower or "failed" in lower:
            self.status_label.configure(fg=self.colors["danger"])
        else:
            self.status_label.configure(fg=self.colors["accent_dark"])

    def _update_task_progress_display(self):
        if self.total_tasks_in_run > 0:
            self.progress_text_var.set(f"Task progress: {self.current_task_index} / {self.total_tasks_in_run}")
            percent = (self.current_task_index / self.total_tasks_in_run) * 100
            self.task_progress["value"] = percent
        else:
            self.progress_text_var.set("Task progress: 0 / 0")
            self.task_progress["value"] = 0

    def _update_participant_progress_display(self):
        if self.total_participants > 0:
            count = len(self.seen_participants)
            self.participant_progress_text_var.set(f"Participant progress: {count} / {self.total_participants}")
            percent = (count / self.total_participants) * 100
            self.participant_progress["value"] = percent
        else:
            if self.current_platform_folder:
                self.participant_progress_text_var.set("Participant progress: tracking not available")
                self.participant_progress["value"] = 0
            else:
                self.participant_progress_text_var.set("Participant progress: 0 / 0")
                self.participant_progress["value"] = 0

    def _set_current_task_context(self, label):
        self.current_task_label = label
        self.task_var.set(f"Current task: {label}")
        self.current_participant = "-"
        self.participant_var.set("Participant: -")
        self.elapsed_var.set("Elapsed: 00:00:00")
        self.seen_participants = set()
        self.total_participants = 0
        self.current_log_pos = 0
        self.current_log_file = None
        self.current_platform_folder = None
        self.current_task_start_time = time.time()

        if label in PLATFORM_INFO:
            folder_name = PLATFORM_INFO[label]["folder"]
            self.current_platform_folder = folder_name
            participant_dirs = self._participant_dirs_for_folder(folder_name)
            self.total_participants = len(participant_dirs)
            self.current_log_file = self._platform_output_dir(folder_name) / "debug_log.txt"

        self._update_participant_progress_display()

    def _process_line_for_progress(self, line):
        found = PARTICIPANT_RE.findall(line)
        if found:
            participant = found[-1].upper()
            self.current_participant = participant
            self.participant_var.set(f"Participant: {participant}")
            self.seen_participants.add(participant)
            self._update_participant_progress_display()

    def _read_new_log_lines(self):
        if not self.current_log_file or not self.current_log_file.exists():
            return

        try:
            with open(self.current_log_file, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(self.current_log_pos)
                new_text = f.read()
                self.current_log_pos = f.tell()
        except Exception:
            return

        if not new_text:
            return

        for raw_line in new_text.splitlines():
            line = raw_line.rstrip()
            if not line:
                continue
            self._process_line_for_progress(line)
            self._log(line)

    def _python_command(self):
        if getattr(sys, "frozen", False):
            return ["py"]
        return [sys.executable]

    def _subprocess_kwargs(self):
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        return kwargs

    def _stream_subprocess_output(self, process):
        try:
            if process.stdout is None:
                return
            for raw_line in iter(process.stdout.readline, ""):
                if raw_line == "":
                    break
                line = raw_line.rstrip()
                if line:
                    self.output_queue.put(("stdout", line))
        except Exception as e:
            self.output_queue.put(("log", f"[WARN] Output reader stopped: {e}"))

    # =========================
    # Buttons / Commands
    # =========================
    def clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def open_base_dir(self):
        try:
            os.startfile(BASE_DIR)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def open_results(self):
        if RESULTS_DIR.exists():
            os.startfile(RESULTS_DIR)
        else:
            messagebox.showwarning("Missing Folder", f"Folder not found:\n{RESULTS_DIR}")

    def open_platform_output(self):
        default_value = self.current_platform_folder if self.current_platform_folder else ""
        choice = simpledialog.askstring(
            "Open Output Folder",
            "Enter platform folder name:\nExamples: aade, efka, govgr, myhealth, telecoms, webbanking",
            initialvalue=default_value,
            parent=self.root
        )

        if not choice:
            return

        folder_name = choice.strip().lower()
        path = self._platform_output_dir(folder_name)

        if path.exists():
            os.startfile(path)
        else:
            messagebox.showwarning("Missing Folder", f"Folder not found:\n{path}")

    def stop_processing(self):
        if not self.running:
            return

        self.stop_requested = True
        self.status_var.set("Stopping...")
        self._update_status_color("Stopping...")
        self._log("\n[USER] Stop requested...")

        if self.current_task_label:
            self.task_status[self.current_task_label] = "Stopping"
            self._update_tree_row(self.current_task_label, status="Stopping")

        if self.current_process:
            try:
                self.current_process.terminate()
                self._log("[INFO] Terminate signal sent to current process.")
            except Exception as e:
                self._log(f"[WARN] Could not terminate process: {e}")

    def run_clean_csv(self):
        script_path = self._check_script_exists("clean_csv.py")
        if not script_path:
            return

        if self.running:
            messagebox.showwarning("Busy", "Another task is already running.")
            return

        folder_name = simpledialog.askstring(
            "Clean CSV Files",
            "Enter platform folder name (example: aade, govgr, efka):",
            parent=self.root
        )

        if not folder_name:
            self._log("Clean CSV cancelled.")
            return

        folder_name = folder_name.strip()
        self._log(f"\n=== Running clean_csv.py for platform: {folder_name} ===")

        try:
            process = subprocess.Popen(
                self._python_command() + [str(script_path), folder_name],
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **self._subprocess_kwargs()
            )

            def reader():
                try:
                    if process.stdout is None:
                        return
                    for raw_line in iter(process.stdout.readline, ""):
                        if raw_line == "":
                            break
                        line = raw_line.rstrip()
                        if line:
                            self.output_queue.put(("log", line))
                except Exception as e:
                    self.output_queue.put(("log", f"[WARN] clean_csv output reader stopped: {e}"))

            threading.Thread(target=reader, daemon=True).start()
            self.status_var.set(f"Running clean_csv for: {folder_name}")
            self._update_status_color(self.status_var.get())

        except Exception as e:
            messagebox.showerror("Error", f"Could not run clean_csv.py\n\n{e}")

    def run_clean_csv_all(self):
        script_path = self._check_script_exists("clean_csv.py")
        if not script_path:
            return

        if self.running:
            messagebox.showwarning("Busy", "Another task is already running.")
            return

        confirm = messagebox.askyesno(
            "Clean ALL CSV Files",
            "This will scan all platform folders under data, normalize participant folder names, extract RAR files, and check/repair all CSV files.\n\nContinue?"
        )

        if not confirm:
            self._log("Clean ALL CSV cancelled.")
            return

        self._log("\n=== Running clean_csv.py for ALL platforms ===")

        try:
            process = subprocess.Popen(
                self._python_command() + [str(script_path), "--all"],
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **self._subprocess_kwargs()
            )

            def reader():
                try:
                    if process.stdout is None:
                        return
                    for raw_line in iter(process.stdout.readline, ""):
                        if raw_line == "":
                            break
                        line = raw_line.rstrip()
                        if line:
                            self.output_queue.put(("log", line))
                except Exception as e:
                    self.output_queue.put(("log", f"[WARN] clean_csv output reader stopped: {e}"))

            threading.Thread(target=reader, daemon=True).start()
            self.status_var.set("Running clean_csv for ALL platforms")
            self._update_status_color(self.status_var.get())

        except Exception as e:
            messagebox.showerror("Error", f"Could not run clean_csv.py\n\n{e}")

    def run_named_script(self, key):
        script_path = self._check_script_exists(SCRIPTS[key])
        if not script_path:
            return
        self._run_sequence([(key, script_path)])

    def run_all_platforms(self):
        tasks = []
        for key in ALL_PLATFORM_KEYS:
            path = self._check_script_exists(SCRIPTS[key])
            if not path:
                return
            tasks.append((key, path))
        self._run_sequence(tasks)

    def run_merge_and_summary(self):
        keys = [
            "Merge Master Dataset",
            "Build Task Summary",
            "Build Platform / Sector Summary",
        ]
        tasks = []
        for key in keys:
            path = self._check_script_exists(SCRIPTS[key])
            if not path:
                return
            tasks.append((key, path))
        self._run_sequence(tasks)

    def run_full_pipeline(self):
        keys = (
            ["Clean CSV Files"]
            + ALL_PLATFORM_KEYS
            + [
                "Merge Master Dataset",
                "Build Task Summary",
                "Build Platform / Sector Summary",
            ]
        )
        tasks = []
        for key in keys:
            path = self._check_script_exists(SCRIPTS[key])
            if not path:
                return
            tasks.append((key, path))
        self._run_sequence(tasks)

    # =========================
    # Pre/Post checks
    # =========================
    def _precheck_task(self, label):
        if label == "Clean CSV Files":
            return True, "Ready"

        if label in PLATFORM_INFO:
            folder_name = PLATFORM_INFO[label]["folder"]
            platform_dir = DATA_DIR / folder_name
            if not platform_dir.exists():
                return False, f"Missing platform folder: {platform_dir}"
            return True, f"Platform folder found: {platform_dir}"

        if label == "Merge Master Dataset":
            if not DATA_DIR.exists():
                return False, f"Missing data directory: {DATA_DIR}"
            return True, "Data directory found"

        if label in ["Build Task Summary", "Build Platform / Sector Summary"]:
            master_file = self._merge_master_file()
            if not master_file.exists():
                return False, f"Missing merged master dataset: {master_file}"
            return True, "Merged master dataset found"

        return True, "Ready"

    def _postcheck_task(self, label):
        if label in PLATFORM_INFO:
            folder_name = PLATFORM_INFO[label]["folder"]
            master_file = self._platform_master_file(folder_name)
            summary_file = self._platform_summary_file(folder_name)

            row_count = 0
            if master_file.exists():
                try:
                    import pandas as pd
                    row_count = len(pd.read_csv(master_file))
                except Exception:
                    row_count = 0

            if row_count == 0:
                return "No Data"

            if summary_file.exists():
                return f"Data OK ({row_count} rows)"
            return f"Master Only ({row_count} rows)"

        if label == "Merge Master Dataset":
            master_file = self._merge_master_file()
            if not master_file.exists():
                return "No Output"
            return "Merged File OK"

        if label == "Build Task Summary":
            summary_file = self._summary_output_file()
            if not summary_file.exists():
                return "No Output"
            return "Summary OK"

        if label == "Build Platform / Sector Summary":
            platform_file = self._platform_sector_summary_file()
            sector_file = self._sector_summary_file()

            if platform_file.exists() and sector_file.exists():
                return "Platform/Sector OK"
            if platform_file.exists() or sector_file.exists():
                return "Partial Output"
            return "No Output"

        return "-"

    # =========================
    # Run sequence
    # =========================
    def _run_sequence(self, tasks):
        if self.running:
            messagebox.showwarning("Busy", "Another task is already running.")
            return

        self.stop_requested = False
        self.total_tasks_in_run = len(tasks)
        self.current_task_index = 0
        self._update_task_progress_display()
        self._reset_task_tree(tasks)
        self._set_running(True)
        self.status_var.set(f"Running: {tasks[0][0]}")
        self._update_status_color(self.status_var.get())
        threading.Thread(target=self._worker, args=(tasks,), daemon=True).start()

    def _worker(self, tasks):
        success = True
        stopped = False

        for idx, (label, script_path) in enumerate(tasks, start=1):
            if self.stop_requested:
                stopped = True
                success = False
                self.output_queue.put(("log", "\n=== Stopped by user ==="))
                break

            self.output_queue.put(("start_task", label, idx - 1, len(tasks)))
            self.output_queue.put(("log", f"\n=== Running: {label} ==="))
            self.output_queue.put(("status", f"Running: {label}"))
            self.output_queue.put(("task_status", label, "Running", "-", "-"))

            ok_precheck, precheck_msg = self._precheck_task(label)
            self.output_queue.put(("log", f"[CHECK] {label}: {precheck_msg}"))

            if not ok_precheck:
                success = False
                self.output_queue.put(("task_status", label, "Precheck Failed", "-", precheck_msg))
                self.output_queue.put(("log", f"=== Precheck failed: {label} ==="))
                break

            task_start = time.time()

            try:
                env = os.environ.copy()
                env["MPLBACKEND"] = "Agg"

                if label == "Clean CSV Files":
                    cmd = self._python_command() + [str(script_path), "--all"]
                else:
                    cmd = self._python_command() + [str(script_path)]

                process = subprocess.Popen(
                    cmd,
                    cwd=str(BASE_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                    **self._subprocess_kwargs()
                )
                self.current_process = process

                reader_thread = threading.Thread(
                    target=self._stream_subprocess_output,
                    args=(process,),
                    daemon=True
                )
                reader_thread.start()

                return_code = process.wait()
                elapsed = self._format_elapsed(time.time() - task_start)
                self.current_process = None

                try:
                    reader_thread.join(timeout=2)
                except Exception:
                    pass

                if self.stop_requested:
                    stopped = True
                    success = False
                    self.output_queue.put(("task_status", label, "Stopped", elapsed, "Stopped by user"))
                    self.output_queue.put(("log", f"=== Stopped during: {label} ==="))
                    break

                if return_code == 0:
                    post_status = self._postcheck_task(label)
                    self.output_queue.put(("task_status", label, "Done", elapsed, post_status))
                    self.output_queue.put(("log", f"=== Completed: {label} ({elapsed}) ==="))
                else:
                    success = False
                    self.output_queue.put(("task_status", label, "Failed", elapsed, f"Exit code {return_code}"))
                    self.output_queue.put(("log", f"=== Failed: {label} (exit code {return_code}) ==="))
                    break

            except Exception as e:
                success = False
                elapsed = self._format_elapsed(time.time() - task_start)
                self.current_process = None
                self.output_queue.put(("task_status", label, "Error", elapsed, str(e)))
                self.output_queue.put(("log", f"=== Error while running {label}: {e} ==="))
                break

        if stopped:
            self.output_queue.put(("finished", False, True))
        else:
            self.output_queue.put(("finished", success, False))

    # =========================
    # UI polling
    # =========================
    def _poll_output(self):
        try:
            while True:
                item = self.output_queue.get_nowait()
                kind = item[0]

                if kind == "log":
                    self._log(item[1])

                elif kind == "stdout":
                    line = item[1]
                    self._process_line_for_progress(line)
                    self._log(line)

                elif kind == "status":
                    self.status_var.set(item[1])
                    self._update_status_color(item[1])

                elif kind == "start_task":
                    label = item[1]
                    current_idx = item[2]
                    total = item[3]
                    self.current_task_index = current_idx
                    self.total_tasks_in_run = total
                    self._set_current_task_context(label)
                    self._update_task_progress_display()

                elif kind == "task_status":
                    label, status, elapsed, data_status = item[1], item[2], item[3], item[4]
                    self.task_status[label] = status
                    self.task_elapsed[label] = elapsed
                    self.task_data_status[label] = data_status
                    self._update_tree_row(label, status=status, elapsed=elapsed, data_status=data_status)

                    done_like = {"Done", "Failed", "Error", "Stopped", "Precheck Failed"}
                    if status in done_like:
                        completed = sum(1 for s in self.task_status.values() if s in done_like)
                        self.current_task_index = completed
                        self._update_task_progress_display()

                elif kind == "finished":
                    success = item[1]
                    stopped = item[2]

                    if stopped:
                        self.status_var.set("Stopped")
                        self._update_status_color("Stopped")
                    elif success:
                        self.status_var.set("Completed")
                        self._update_status_color("Completed")
                    else:
                        self.status_var.set("Failed")
                        self._update_status_color("Failed")

                    self.current_task_index = self.total_tasks_in_run
                    self._update_task_progress_display()
                    self._set_running(False)

        except queue.Empty:
            pass

        self.root.after(100, self._poll_output)

    def _poll_runtime_state(self):
        if self.running:
            if self.current_task_start_time is not None:
                elapsed = self._format_elapsed(time.time() - self.current_task_start_time)
                self.elapsed_var.set(f"Elapsed: {elapsed}")

            self._read_new_log_lines()

        self.root.after(350, self._poll_runtime_state)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()