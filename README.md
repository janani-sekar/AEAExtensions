# AEA Extensions: Automated Empirical Economics Analysis Agent

An AI agent that automatically generates and executes empirical economics analyses on replication packages.

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd AEAExtensions
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
# Create a .env file
echo "OPENAI_API_KEY=sk-xxxxxxxxxxxxx" > .env
```

## Quick Start

### Running on AEA Replication Packages

1. **Download a replication package manually**:
   - Go to [AEA Data and Code Repository](https://www.aeaweb.org/journals/datasets)
   - Find a paper (e.g., [Insecurity and Firm Displacement](https://www.openicpsr.org/openicpsr/project/208787/version/V1/view))
   - Click "Download this project" (requires free ICPSR account)
   - Unzip to a local directory:
     ```bash
     mkdir -p aea_packages
     unzip 208787_V1.zip -d aea_packages/208787_V1
     ```

2. **Catalog the data** (optional, to see what files were discovered):
```bash
python run.py --data-dir aea_packages/208787_V1 \
  --paper-pdf aea_packages/208787_V1/paper.pdf \
  --analysis-name my_analysis \
  --catalog-only
```

3. **Run the analysis**:
```bash
python run.py --data-dir aea_packages/208787_V1 \
  --paper-pdf aea_packages/208787_V1/paper.pdf \
  --analysis-name my_analysis \
  --num-analyses 1 \
  --max-iterations 2 \
  --max-fix-attempts 2
```

### Running on a Single Dataset

If you have a single CSV/Parquet/Stata file:

```bash
python run.py --data-path path/to/data.csv \
  --paper-pdf path/to/paper.pdf \
  --analysis-name my_analysis \
  --num-analyses 1 \
  --max-iterations 2
```

Or with a text summary instead of PDF:
```bash
python run.py --data-path path/to/data.csv \
  --paper-path path/to/summary.txt \
  --analysis-name my_analysis \
  --num-analyses 1 \
  --max-iterations 2
```

## What It Does

1. **Data Discovery** (if using `--data-dir`):
   - Scans directory for tabular files (CSV, Parquet, Feather, Stata)
   - Builds a catalog with column metadata
   - Selects primary analysis file using heuristics (panel structure, treatment indicators)

2. **Paper Understanding**:
   - Extracts text from PDF (if `--paper-pdf` provided)
   - Summarizes research question, variables, identification strategy
   - Uses Deep Research to gather additional context

3. **Analysis Generation**:
   - Proposes novel analyses distinct from the paper
   - Generates Python code (pandas, statsmodels, linearmodels)
   - Executes in Jupyter notebooks with auto-fixing on errors
   - Interprets results and suggests next steps

4. **Outputs**:
   - Notebooks: `outputs/<analysis_name>_<timestamp>/*.ipynb`
   - Logs: `logs/<analysis_name>_log_<timestamp>.log`
   - Catalog: `logs/dataset_catalog_<timestamp>.json` (if using `--data-dir`)

## Command-Line Options

### Data Input
- `--data-path`: Single tabular file (CSV/Parquet/Feather/Stata)
- `--data-dir`: Directory with multiple files (replication package)
- `--data-glob`: File patterns to discover (default: `*.csv,*.parquet,*.feather,*.dta`)
- `--primary-file`: Override primary file selection within `--data-dir`

### Paper Input
- `--paper-pdf`: PDF of research paper (auto-extracts and summarizes)
- `--paper-path`: Pre-written text summary

### Analysis Control
- `--analysis-name`: Name for output files (default: `covid19`)
- `--num-analyses`: Number of distinct analyses (default: `8`)
- `--max-iterations`: Steps per analysis (default: `6`)
- `--max-fix-attempts`: Auto-fix attempts on errors (default: `3`)
- `--model-name`: OpenAI model (default: `o3-mini`)

### Optional Features
- `--no-self-critique`: Disable self-critique loop
- `--no-vlm`: Disable vision language model for plots
- `--no-documentation`: Disable doc-assisted error fixing
- `--log-prompts`: Save all prompts to logs
- `--catalog-only`: Build catalog and exit (no analysis)

### Schema Hints (Optional)
- `--outcome`: Outcome variable column name
- `--treatment`: Treatment indicator column name
- `--time-var`: Time variable for panels/event studies
- `--unit-var`: Unit identifier for panels
- `--cluster-se`: Column for clustering standard errors

## Example Workflow

```bash
# 1. Download replication package from openICPSR
# (manual download required; see above)

# 2. Quick test (1 analysis, 1 iteration, no recovery)
python run.py --data-dir aea_packages/208787_V1 \
  --paper-pdf aea_packages/208787_V1/paper.pdf \
  --analysis-name quick_test \
  --num-analyses 1 --max-iterations 1 --max-fix-attempts 0 \
  --no-self-critique --no-vlm --no-documentation

# 3. Full run with all features
python run.py --data-dir aea_packages/208787_V1 \
  --paper-pdf aea_packages/208787_V1/paper.pdf \
  --analysis-name full_run \
  --num-analyses 2 --max-iterations 3 --max-fix-attempts 2 \
  --log-prompts
```

## Included Example

A small demo dataset and paper summary are included:

```bash
python run.py --data-path example/econ_demo.csv \
  --paper-path example/econ_paper_summary.txt \
  --analysis-name demo \
  --num-analyses 1 --max-iterations 2
```

## Requirements

- Python 3.8+
- OpenAI API key
- See `requirements.txt` for package dependencies

## License

See `LICENSE` file.
