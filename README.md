# EquipValue: Heavy Equipment Price Intelligence

EquipValue is a student-friendly decision-support project for estimating the
selling price of used heavy equipment from auction date, age, usage, location,
and technical attributes. It extends an IIT Madras/Kaggle competition solution
into a reproducible Python, SQLite, R, classical-ML, PyTorch, and Streamlit
decision-support pipeline.

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

### 10: business recommendation layer

`src/predictor.py` turns a raw model score into an expected selling-price
estimate, empirical error range, per-row LightGBM feature contributions,
similar-category historical average, unfamiliar-input warnings, and a human
inspection disclaimer. `src/build_business_context.py` derives aggregate
category statistics and the error range from time-aware out-of-fold evidence.

The output supports reserve-price planning, buyer budgeting, inventory
valuation, and manual-review prioritization. It never makes an automatic
purchase or sale recommendation.

### 11: tested Streamlit application

`app/streamlit_app.py` provides a single-equipment form, chunked CSV upload,
feature explanations, limitations, invalid-row reporting, and downloadable
predictions. The app has passed automated and local interactive smoke tests.

**Deployment status:** tested locally, not publicly deployed. No live URL is
claimed because the trained artifacts are derived from access-controlled data
and remain ignored until redistribution/deployment permission is confirmed.

### 12: scale-aware batch inference

`src/batch_predict.py` streams CSV chunks, writes valid predictions separately
from invalid rows, preserves row counts, and benchmarks runtime and memory.
This is described as scale-aware local inference, not an Amazon-scale or
distributed production system.

### 13: tests and portfolio packaging

The test suite covers feature schema, target leakage, unknown categories,
invalid dates/hours, model reload, non-negative predictions, batch row counts,
benchmark evidence, and Streamlit startup/interactive prediction.

### 14-17: decision-science extensions

`src/market_forecasting.py` aggregates auction history into monthly median
prices, holds out the latest 12 months, and compares three transparent market
forecast baselines: last observed price, a 12-month seasonal naive model, and a
linear log-price trend. The selected model is used only for a short historical
scenario after the dataset end date. It is not presented as a current-market
forecast because the authorised history ends in 2013.

`r_analysis/equipment_price_analysis.R` now adds two formal hypothesis tests:

- one-way ANOVA of log price across represented equipment categories, with
  eta-squared effect size;
- Welch's t-test of High- versus Low-utilization log price, with Cohen's d.

Large samples can make tiny differences statistically significant, so the
report includes effect sizes and never turns a p-value into a causal claim.

`src/causal_sensitivity.py` uses propensity-score weighting to compare High-
and Low-utilization equipment after balancing observed age, sale year, category,
and region. It reports overlap, effective sample size, balance diagnostics, an
adjusted association, assumptions, and unmeasured-confounding limitations. This
is an observational sensitivity analysis, not proof of causation.

`src/inventory_optimization.py` adds a classic EOQ and reorder-point scenario by
equipment category. Auction transaction volume is only a market-activity proxy;
ordering cost, holding rate, lead time, and service target come from an explicit
user-editable assumptions file. The output is decision support, not an automatic
purchase instruction.

## Architecture

```text
Clean auction data ──> fold-safe features ──> LightGBM model
        │                       │                    │
        ├──> SQL/R evidence     └──> fitted preprocessing
        │                                            │
        └──> OOF errors + category aggregates ───────┤
                                                     v
                                           PricePredictor service
                                              /             \
                                     Streamlit app      Chunked CSV CLI

Monthly aggregates ──> forecast backtest ──> historical market scenario
R group analysis ─────> ANOVA / Welch test ─> p-values + effect sizes
Observed confounders ─> propensity weighting -> causal sensitivity report
Category activity + user assumptions ───────> EOQ / reorder scenario
```

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

### Decision-science extension results

The monthly market backtest uses 248 training months and the latest 12 months
as validation. Seasonal-naive forecasting achieved the lowest validation RMSE
among the three simple candidates at **4,021**, compared with **4,243** for the
last-observed baseline and **4,220** for a linear log trend. Its MAE was not the
best, so the report preserves all metrics instead of presenting one model as
universally superior.

The R analysis produced two formal tests. Category ANOVA returned eta-squared
of approximately **0.053**, while the unadjusted High-versus-Low utilization
comparison returned Cohen's d of approximately **0.916**. With more than one
hundred thousand records, both p-values were below ordinary printed precision;
effect sizes and assumptions are therefore essential for interpretation.

The propensity-weighted utilization sensitivity analysis retained **25,360**
High/Low rows after overlap trimming. Age imbalance fell from an absolute
standardized difference of **0.458** to **0.094**, and sale-year imbalance fell
from **0.066** to **0.003**. The adjusted association remains vulnerable to
unmeasured condition, damage, seller behaviour, and local-demand variables, so
it is not promoted as a proven causal effect.

The EOQ/reorder output is reproducible with the example assumptions, but those
numbers are illustrative. Real cost, demand, and lead-time inputs must replace
the example before the result can support a dealer decision.

### Empirical range and batch benchmark

The displayed price range uses the 90th percentile absolute log error from
129,968 time-aware LightGBM out-of-fold predictions. It covered 90.0% of those
historical OOF rows by construction; it is not a guaranteed confidence interval.

The complete prediction path—including validation, model inference, per-row
feature contributions, warnings, and result formatting—was benchmarked locally
with 1,000-row chunks:

| Requested rows | Valid predictions | Invalid rows | Runtime | Rows/second | Python peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1,000 | 0 | 3.2 s | 310 | 22.6 MB |
| 10,000 | 9,480 | 520 | 35.4 s | 282 | 23.1 MB |
| 50,000 | 45,234 | 4,766 | 183.4 s | 273 | 23.8 MB |

Invalid benchmark rows contain data-quality failures such as impossible raw
manufacture years and are reported rather than silently scored. Near-flat traced
Python memory demonstrates bounded chunk processing on this machine; results are
not cloud-throughput or production-scale claims.

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
│   ├── analyze_errors.py
│   ├── build_business_context.py
│   ├── predictor.py
│   └── batch_predict.py
├── app/
│   └── streamlit_app.py
├── examples/
│   └── batch_input_template.csv
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
python -m src.build_business_context
python -m src.market_forecasting
python -m src.causal_sensitivity
python -m src.inventory_optimization
python -m src.batch_predict --benchmark
streamlit run app/streamlit_app.py
```

For a real CSV batch:

```bash
python -m src.batch_predict \
  --input path/to/equipment.csv \
  --output predictions/equipment_predictions.csv \
  --invalid-output predictions/invalid_rows.csv \
  --chunk-size 1000
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
reports/market_forecast_backtest.csv
reports/market_forecast_metrics.json
reports/market_forecast_backtest.png
reports/market_forecast_future.csv
reports/r_hypothesis_tests.csv
reports/causal_sensitivity.json
reports/inventory_optimization.csv
reports/inventory_optimization.json
reports/batch_benchmark.json
models/lightgbm_model.txt
models/pytorch_mlp.pt
models/business_context.json
```

Generated rows, aggregates, and model binaries are ignored. Public code and
tests reproduce them for an authorised dataset holder.

## Tests

```bash
pytest -q
```

Tests use small synthetic fixtures. They verify data cleaning, SQL/R contracts,
row features, training-only frequency mappings, strict future-date folds,
AssetID overlap detection, PyTorch save/load, business recommendations,
unfamiliar inputs, target exclusion, chunked row counts, and Streamlit startup.
The current local run contains 23 passing tests. PyTorch and Streamlit tests are
dependency-gated and skip when those optional packages are unavailable in the
active Python environment. Synthetic test values are never presented as
competition results.

## Limitations

- Training history ends in April 2013, so modern-market predictions are a major
  distribution shift and trigger a warning.
- The empirical range reflects past OOF errors and is not a calibrated guarantee.
- The model cannot inspect mechanical condition, damage, title history, local
  demand, attachments, or auction-specific negotiation.
- LightGBM contributions explain model behaviour, not real-world causation.
- Forecast outputs stop immediately after the 2013 dataset and are not current-
  market guidance.
- Causal sensitivity depends on measured-confounder and overlap assumptions;
  unmeasured condition and market variables remain.
- EOQ and reorder results use an auction-activity proxy plus illustrative costs,
  not verified dealer demand or accounting inputs.
- The application is local software until a public URL and artifact-permission
  path are both verified.

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
