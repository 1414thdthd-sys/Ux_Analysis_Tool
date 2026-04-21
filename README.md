UX Analysis Tool
Overview

This repository contains the implementation developed for the thesis:

"Evaluation of Usability and User Behavior in Public and Private Sector Digital Services in Greece Using Multimodal Biometric Measurements"

The system processes multimodal user interaction and biometric data (eye tracking, physiological signals, behavioral logs) to evaluate usability, cognitive load, and user performance across digital platforms.

Author

Theodoros Makris
University of the Aegean
School of Engineering
Department of Information and Communication Systems Engineering

Project Structure

UX_Analysis_Tool/
│
├── gui_runner.py # Main GUI interface
├── UX_Analysis_Tool.exe # Executable version (Windows)
│
├── spython/ # Analysis scripts
│
└── results/ # Generated outputs (ignored in repository)

Pipeline Overview

The system follows a structured processing pipeline:

Data Cleaning
Standardizes raw participant files
Platform Analysis
Extracts metrics per platform (govgr, aade, efka, etc.)
Generates:
Task-level metrics
Physiological indicators
Graphs
Aggregation
Merges all platform datasets
Produces global summaries and rankings
Subjective Analysis
Processes SUS and NASA-TLX questionnaires
Main Scripts
gui_runner.py → GUI-based pipeline execution
clean_csv.py → Data cleaning
govgr.py, aade.py, efka.py → Platform-level analysis
merge_master_datasets.py → Dataset merging
build_task_summary_and_rankings.py → Task-level aggregation
build_platform_sector_summaries.py → Platform/sector comparison
build_subjective_graphs.py → SUS & NASA-TLX analysis
Requirements
Python 3.10+ (3.11 recommended)

Required Python packages:

pandas
numpy
matplotlib

Install dependencies:

pip install pandas numpy matplotlib

How to Run
Option A — GUI (Recommended)
Place the folder UX_Analysis_Tool on Desktop
Run:
UX_Analysis_Tool.exe
Choose:
Manual Analysis (step-by-step)
Full Pipeline (automatic)
Option B — Python Scripts (Advanced)

Run step-by-step:

python clean_csv.py
python govgr.py
python merge_master_datasets.py
python build_task_summary_and_rankings.py
python build_subjective_graphs.py

Outputs
Platform-level outputs

Generated in:
data/<platform>/analysis_output/

Contains:

Platform master dataset
Task summary
Graphs (cognitive load, HR, EDA, gaze, multimodal)
Global outputs

Generated in:
results/

Contains:

master_dataset.csv → combined dataset from all platforms
task_summary.csv → aggregated metrics per task
task_rankings.csv → task evaluation and ranking
Important Notes
Running the pipeline overwrites previous outputs
Always rerun the full pipeline after adding new data
Ensure consistent file naming and structure across inputs
Data Availability

No datasets are included in this repository.
Data must be provided separately.

License

This project is licensed under the MIT License.
