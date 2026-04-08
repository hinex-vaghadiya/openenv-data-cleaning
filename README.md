---
title: Data Cleaning Env
emoji: 👀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

#  Data Cleaning Environment

> An OpenEnv-compliant RL environment where AI agents learn to clean messy tabular datasets  a skill data engineers spend **80% of their time** on in the real world.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

---

##  Motivation

Data cleaning is **the most time-consuming part** of any data pipeline. According to industry surveys, data professionals spend 60-80% of their time finding and fixing data quality issues. This environment simulates that real-world task, providing a structured training ground for AI agents to learn data cleaning skills.

Unlike toy environments, this one models genuine data quality issues that professionals encounter daily:
- Duplicate records from ETL pipeline failures
- Missing values from incomplete data ingestion
- Format inconsistencies across data sources
- Statistical outliers from sensor errors or data entry mistakes
- Schema drift from evolving data sources
- Typos and encoding issues in categorical data

---

##  Environment Design

### Action Space

The agent can perform **9 types of cleaning actions**:

| Action | Description | Parameters |
|--------|-------------|------------|
| `remove_duplicates` | Remove duplicate rows | `column` (optional subset) |
| `fill_missing` | Handle missing values | `column`, `strategy` (drop/fill_value/mean/mode), `fill_value` |
| `standardize_format` | Standardize data formats | `column`, `format` (iso_date/phone_standard/lowercase/titlecase/strip_whitespace) |
| `fix_outliers` | Handle statistical outliers | `column`, `strategy` (clip/remove), `lower_bound`, `upper_bound` |
| `rename_column` | Rename a column | `column`, `new_name` |
| `drop_column` | Drop irrelevant column | `column` |
| `correct_typos` | Fix categorical inconsistencies | `column`, `mapping` (oldnew dict) |
| `convert_type` | Convert column data type | `column`, `target_type` (float/int/str), `strip_chars` |
| `submit` | Submit for final grading |  |

### Observation Space

Each observation contains:
- **Dataset summary**: row count, column count, column names, types
- **Quality metrics**: missing value counts per column, duplicate count
- **Column stats**: mean, std, min, max for numeric columns
- **Sample data**: first 5 rows as key-value dictionaries
- **Detected issues**: auto-detected data quality problems
- **Action feedback**: success/failure and message from last action
- **Task context**: task ID, description, step count, quality score

### Reward Function

The reward function provides **continuous partial-progress signals**:

1. **Step reward** = ` quality_score - 0.005` (improvement minus small time penalty)
2. **Submit reward** = `final_quality_score` (0.01.0)
3. **Timeout penalty** = `quality_score  0.8` (20% penalty for exhausting steps)

Quality score is a **weighted multi-dimensional metric** computed across:
- Row count accuracy (no lost or extra rows)
- Missing value reduction
- Duplicate elimination
- Data type correctness
- Format consistency
- Outlier handling
- Schema correctness
- Irrelevant column removal

---

##  Tasks

### Task 1: Basic Data Cleaning (Easy)
- **Dataset**: 30 employee records + 5 duplicates = 35 rows
- **Issues**: Duplicate rows, missing emails/salaries/departments, salary as string
- **Max steps**: 15
- **Expected score for good agent**: 0.851.0

### Task 2: Format Standardization (Medium)
- **Dataset**: 40 customer records + 4 duplicates = 44 rows
- **Issues**: Mixed date formats, inconsistent phone formats, state abbreviation vs full name, category typos, duplicates, missing values
- **Max steps**: 25
- **Expected score for good agent**: 0.700.90

### Task 3: Complex Multi-Issue Cleaning (Hard)
- **Dataset**: 60 sales transactions + 6 duplicates + 3 irrelevant columns = 66 rows, 12 columns
- **Issues**: Revenue outliers, negative quantities, future dates, mixed date formats, encoding issues, status typos, irrelevant columns, revenue as "$X.XX" strings, duplicates, missing values, schema inconsistency (unit_price  unit_cost)
- **Max steps**: 35
- **Expected score for good agent**: 0.500.80

---

##  Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)

### Local Development

```bash
# Clone the repository
git clone <repo-url>
cd data-cleaning-env

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
python server/app.py
```

### Docker

```bash
# Build the image
docker build -t data-cleaning-env .

# Run the container
docker run -d -p 7860:7860 data-cleaning-env
```

### Test the endpoints

```bash
# Health check
curl http://localhost:7860/health

# Reset environment (easy task)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "task_id": "task_easy"}'

# Execute a cleaning action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "remove_duplicates", "column": null, "params": {}}}'

# Get current state
curl http://localhost:7860/state
```

### Run the Baseline Inference

```bash
# Set environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="your-key-here"
export ENV_URL="http://localhost:7860"

# Run inference
python inference.py
```

---

##  Baseline Scores

| Task | GPT-4o-mini Score | Steps Used |
|------|------------------|------------|
| task_easy | ~0.85 | 58 |
| task_medium | ~0.65 | 1015 |
| task_hard | ~0.45 | 1525 |

*Scores are deterministic with seed=42. Better models and strategies can significantly improve these scores.*

---

##  Project Structure

```
data-cleaning-env/
 openenv.yaml          # OpenEnv manifest
 models.py             # Typed Action/Observation/State models
 server/
    app.py            # FastAPI server with OpenEnv routes
    environment.py    # Core environment logic + graders
 inference.py          # Baseline inference script
 Dockerfile            # Container definition
 requirements.txt      # Python dependencies
 pyproject.toml        # Package metadata
 README.md             # This file
```

---

##  Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 7860 | Server port |
| `HOST` | 0.0.0.0 | Bind address |
| `API_BASE_URL` | https://api.openai.com/v1 | LLM API endpoint |
| `MODEL_NAME` | gpt-4o-mini | Model identifier |
| `HF_TOKEN` |  | API key |
| `OPENAI_API_KEY` |  | Alternative API key |
| `ENV_URL` | http://localhost:7860 | Environment server URL |

---

##  License

BSD 3-Clause License
