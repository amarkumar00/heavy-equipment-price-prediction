# EquipValue: Heavy Equipment Price Intelligence

EquipValue is a student-friendly decision-support project for estimating the
selling price of used heavy equipment from auction, age, usage, location, and
technical attributes. It extends an IIT Madras/Kaggle competition solution into
a reproducible Python, SQLite, and R analysis pipeline.

The original competition notebook is preserved at
`notebooks/original_competition_solution.ipynb`. The refactor does not overwrite
or retroactively change its reported competition evidence.

## Provenance and academic-integrity boundary

The preserved notebook is the student's original competition work. The Python,
SQL, R, testing, and repository refactor around it is a post-competition
portfolio extension developed with AI coding assistance. It must not be
resubmitted as original IIT Madras coursework or as a competition entry. The
student should read, run, modify, and be able to explain every new module before
using the project in an interview.

## Problem statement

Buyers, sellers, and auction teams need a consistent starting point for valuing
used equipment. This project cleans historical auction records, analyses price
drivers, compares model-ready features, and prepares the repository for later
machine-learning, PyTorch, and deployment milestones.

The system is decision support, not a certified appraisal. Statistical
association is not described as causation, and predicted prices must not replace
physical inspection or expert judgement.

## Why this problem

- It is a real mixed-type regression problem with dates, numeric measurements,
  missing values, and high-cardinality categories.
- Equipment age and operational usage create understandable feature-engineering
  and data-quality questions.
- SQL answers business questions, R provides transparent statistical evidence,
  and Python supports a reproducible modelling pipeline.
- The author has already studied the original notebook, making the extension
  explainable in an interview instead of being a keyword-only rewrite.

## Current milestone scope

### Milestone 1: repository and data contract

- Preserves the original competition notebook.
- Documents the authorised local dataset layout.
- Excludes raw, processed, database, report, and model artifacts from Git.
- Provides actionable errors when data is missing.

### Milestone 2: reproducible cleaning

`src/clean_data.py` parses transaction dates, removes exact duplicates, reports
duplicate identifiers, excludes invalid training targets, treats impossible
manufacture years as missing, inspects operational hours, creates model-ready
features, and writes a JSON data-quality report.

### Milestone 3: SQLite business analysis

`src/build_database.py` normalizes cleaned rows into `equipment_attributes` and
`equipment_transactions`. The SQL analysis answers:

1. How does average price vary by equipment age?
2. Which equipment categories have the highest median price?
3. How does average price vary by region?
4. How do utilization tiers compare?
5. How does price vary across operational-hour bands?
6. Which categories contain the most transactions?

### Milestone 4: R statistical analysis

The package-free base-R script calculates descriptive statistics, correlations,
group summaries, a small log-price regression, and three visual reports. Its
outputs explicitly describe associations rather than causal effects.

## Why R rather than SAS or MATLAB

The role wording presents R, SAS, and MATLAB as alternative statistical tools.
R was selected because it is free, reproducible, strong for statistics and
visualisation, and natural for a CSV/SQL business-analysis workflow.

- SAS is powerful in regulated enterprise environments but is proprietary and
  licence-dependent, which makes a public student project harder to reproduce.
- MATLAB is excellent for engineering simulation, numerical computing, and
  signal processing, but it is also proprietary and less natural for this
  tabular auction-analysis workflow.

The project uses one tool deeply enough to explain instead of adding several
tools only as resume keywords.

## Repository structure

```text
.
├── .github/workflows/notebook-check.yml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   └── original_competition_solution.ipynb
├── sql/
│   ├── create_tables.sql
│   └── analysis_queries.sql
├── r_analysis/
│   └── equipment_price_analysis.R
├── src/
│   ├── clean_data.py
│   ├── build_database.py
│   └── run_sql_analysis.py
├── app/
├── models/
├── reports/
├── tests/
├── requirements.txt
└── README.md
```

The empty `app/` folder is reserved for the deployment milestone. The model
folder contains only its artifact policy until training milestones begin.

## Dataset setup

Join the authorised competition and place these files locally:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

See `data/README.md` for the schema contract. The dataset and all row-level
derived files remain untracked because competition access does not automatically
grant redistribution permission.

## Run milestones 1-4

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m src.clean_data
python -m src.build_database
python -m src.run_sql_analysis
Rscript r_analysis/equipment_price_analysis.R
```

Expected local outputs:

```text
data/processed/clean_train.csv
data/equipvalue.db
reports/data_quality_report.json
reports/sql_business_summary.csv
reports/r_statistical_summary.csv
reports/r_group_summary.csv
reports/price_by_category.png
reports/age_vs_price.png
reports/regression_diagnostics.png
```

## Tests

```bash
pytest -q
```

Tests use a small synthetic fixture. This checks the code without presenting
synthetic values as competition results. The suite covers cleaning rules,
missing-file errors, database row counts, all six SQL analyses, and the complete
R report contract when R is installed.

## Original competition evidence

The original executed competition notebook produced:

| Experiment | RMSLE |
| --- | ---: |
| LightGBM baseline holdout | 0.196867 |
| Tuned LightGBM holdout | 0.196195 |
| Public competition submission | 0.19236 |

Lower RMSLE is better. These values belong to the original executed notebook.
The public notebook was cleaned for review and has no embedded outputs, so the
repository does not claim that those metrics were reproduced by the new scripts.

The notebook already demonstrates Ridge and Decision Tree baselines, feature
engineering, leakage-safe training-only frequency features, LightGBM tuning,
early stopping, and submission validation. Future milestones will move modelling
into scripts, add time-aware cross-validation, compare a small PyTorch MLP, and
create a tested user-facing application.

## Links

- [Public Kaggle notebook](https://www.kaggle.com/code/amar23f2000465/heavy-equipment-price-prediction-portfolio)
- [Competition page](https://www.kaggle.com/competitions/heavy-equipment-selling-price-prediction-challenge)
- [GitHub profile](https://github.com/amarkumar00)
