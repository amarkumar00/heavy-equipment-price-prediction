# EquipValue: Heavy Equipment Price Intelligence

EquipValue is a student-friendly decision-support project for estimating the
selling price of used heavy equipment from auction date, age, usage, location,
and technical attributes. It extends an IIT Madras/Kaggle competition solution
into a reproducible Python, SQLite, R, classical-ML, and PyTorch pipeline.

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
used equipment. EquipValue cleans historical auction records, analyses price
drivers, compares models on genuinely later auctions, and identifies the kinds
of records on which predictions are less reliable.

The system is decision support, not a certified appraisal. Statistical
association is not described as causation, and predicted prices must not replace
physical inspection or expert judgement.

## Why this problem

- It is a real mixed-type regression problem with dates, numeric measurements,
  missing values, and high-cardinality categories.
- Equipment age and operational usage create understandable feature-engineering
  and data-quality questions.
- SQL answers business questions, R provides transparent statistical evidence,
  Python supports reproducible modelling, and PyTorch tests whether a small
  neural network actually adds value.
- The author has already studied the original notebook, making the extension
  explainable in an interview instead of being a keyword-only rewrite.

## Implemented milestones

### 1-4: data, SQL, and R foundation

- Preserves the original competition notebook and keeps access-controlled data
  out of Git.
- Cleans dates, targets, manufacture years, operational hours, duplicates, and
  missing values in `src/clean_data.py`.
- Builds normalized SQLite tables and answers six business questions with SQL.
- Uses base R for descriptive statistics, correlations, a small log-price
  regression, group comparisons, and visual diagnostics.

### 5: leakage-safe Python features

`src/features.py` creates sale date, equipment age, log-hours, hours-per-year,
missingness, utilization, scale, description-length, and category-frequency
features. Frequency mappings are fitted separately inside each training fold;
an unseen validation category receives frequency zero.

### 6: proper time validation

`src/validation.py` sorts unique transaction dates and creates three expanding
windows. Every validation auction is later than its training data, and all rows
from the same date remain together. A separate AssetID report shows whether a
repeated machine appeared in training, so repeated equipment cannot silently
inflate the headline score.

### 7: classical model comparison

`src/train_classical_models.py` evaluates a median baseline, Ridge Regression,
Decision Tree, and LightGBM on the same features and folds. Models learn
`log1p(price)` and are compared with RMSLE, MAE, RMSE, and R-squared. LightGBM is
then retrained on all authorised history and saved with its fitted preprocessing
artifacts.

### 8: simple PyTorch deep learning

`src/train_pytorch.py` implements:

```text
Input -> Dense 128 + ReLU -> Dropout -> Dense 64 + ReLU -> Dense 1
```

Numeric features are scaled, categorical features are encoded, and log-price is
standardized with training-fold statistics. Early stopping monitors untouched
future rows. A typical best epoch count from cross-validation is used to retrain
the saved MLP on all authorised history.

### 9: error and model analysis

`src/analyze_errors.py` compares matching out-of-fold LightGBM and PyTorch
predictions for equipment age, observed price band, region, category, missing
hours, and previously seen versus unseen machines. These are descriptive failure
patterns, not proven root causes or causal effects.

## Verified local results

The complete 138,701-row authorised training file was used. Three expanding
time folds produced the following mean validation results:

| Model | Mean RMSLE | Mean MAE | Mean RMSE | Mean R-squared |
| --- | ---: | ---: | ---: | ---: |
| LightGBM | **0.3493** | **10,478** | **14,576** | **0.688** |
| PyTorch MLP | 0.3840 | 12,077 | 16,835 | 0.583 |
| Ridge Regression | 0.4284 | 13,507 | 21,056 | 0.349 |
| Decision Tree | 0.4359 | 12,801 | 17,720 | 0.540 |
| Median baseline | 0.6734 | 20,669 | 26,709 | -0.039 |

On the 129,968 rows for which both strong models have matching out-of-fold
predictions, LightGBM RMSLE is 0.3359 and PyTorch RMSLE is 0.3849. This is a
useful negative result: for this tabular dataset, the small neural network is
valid but does not beat gradient-boosted trees.

Only 2.9%-4.4% of validation rows across the three folds contained an AssetID
already seen in training. LightGBM scored similarly on seen and unseen machines
(0.3360 versus 0.3359 RMSLE), so repeated assets did not explain its overall
result. Missing/unknown equipment age is a visibly harder segment; this is a
failure pattern to investigate, not proof of causation.

These new time-aware scores are intentionally separate from the original
competition evidence below.

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
├── data/                         # local raw and processed data (ignored)
├── notebooks/
│   └── original_competition_solution.ipynb
├── sql/
├── r_analysis/
├── src/
│   ├── clean_data.py
│   ├── build_database.py
│   ├── run_sql_analysis.py
│   ├── features.py
│   ├── prepare_data.py
│   ├── validation.py
│   ├── metrics.py
│   ├── train_classical_models.py
│   ├── train_pytorch.py
│   └── analyze_errors.py
├── models/                       # generated model artifacts (ignored)
├── reports/                      # generated reports (ignored)
└── tests/
```

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

## Run the complete pipeline

Use Python 3.11-3.13 for the widest PyTorch wheel compatibility:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m src.clean_data
python -m src.build_database
python -m src.run_sql_analysis
Rscript r_analysis/equipment_price_analysis.R
python -m src.prepare_data
python -m src.train_classical_models
python -m src.train_pytorch
python -m src.analyze_errors
```

Main generated outputs:

```text
data/processed/clean_train.csv
data/processed/model_data.csv
data/equipvalue.db
reports/model_comparison.csv
reports/feature_importance.csv
reports/asset_overlap_report.csv
reports/pytorch_metrics.json
reports/pytorch_training_curve.png
reports/model_error_summary.csv
reports/error_analysis.csv
reports/error_analysis.png
models/lightgbm_model.txt
models/pytorch_mlp.pt
```

Generated rows, aggregates, and model binaries are ignored. Public code and
tests reproduce them for an authorised dataset holder.

## Tests

```bash
pytest -q
```

Tests use small synthetic fixtures. They verify data cleaning, SQL/R contracts,
row features, training-only frequency mappings, strict future-date folds,
AssetID overlap detection, and the PyTorch architecture/save-load contract.
Synthetic test values are never presented as competition results.

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
The newer scores are higher partly because validation is deliberately harder:
models train on earlier auctions and must predict later market periods.

## Links

- [Public Kaggle notebook](https://www.kaggle.com/code/amar23f2000465/heavy-equipment-price-prediction-portfolio)
- [Competition page](https://www.kaggle.com/competitions/heavy-equipment-selling-price-prediction-challenge)
- [GitHub profile](https://github.com/amarkumar00)
