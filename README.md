# Heavy Equipment Price Prediction

An end-to-end regression notebook for predicting the selling price of used heavy equipment from mixed numerical and categorical auction data.

This portfolio copy focuses on the reasoning behind the pipeline: exploratory analysis, leakage-safe preprocessing, model comparison, LightGBM tuning, and submission validation.

## Results

The original executed competition notebook produced these verified results:

| Experiment | RMSLE |
| --- | ---: |
| LightGBM baseline holdout | 0.196867 |
| Tuned LightGBM holdout | 0.196195 |
| Public competition submission | 0.19236 |

Lower RMSLE is better. The values above belong to the original executed notebook; this simplified portfolio copy was cleaned for public review and has not been re-executed without the competition data.

## What the notebook demonstrates

- EDA for target skew, missing values, duplicates, and feature relationships
- `log1p` target modelling so RMSE in log space matches the RMSLE objective
- A shared holdout split for fair model comparison
- `Pipeline` and `ColumnTransformer` preprocessing for Ridge and Decision Tree baselines
- Date, age, usage, ordinal, missingness, and categorical-frequency features
- Leakage-safe fitting of frequency features on training rows only
- Native categorical handling and early stopping with LightGBM
- Six visible LightGBM configurations evaluated under the same validation rule
- Assertions for submission schema, row order, uniqueness, finite values, and non-negative predictions

## Repository structure

```text
.
├── notebooks/
│   └── heavy_equipment_price_prediction.ipynb
├── .github/workflows/notebook-check.yml
├── requirements.txt
└── README.md
```

## Run locally

1. Join the Kaggle competition and download its authorised files.
2. Place `train.csv`, `test.csv`, and `sample_submission.csv` in one local data folder.
3. Update the data path in the notebook for that folder.
4. Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

The competition dataset is intentionally not committed because access and redistribution are governed by the competition rules.

## Links

- [View the public notebook on Kaggle](https://www.kaggle.com/code/amar23f2000465/heavy-equipment-price-prediction-portfolio)
- [Kaggle profile](https://www.kaggle.com/amar23f2000465)
- [Competition page](https://www.kaggle.com/competitions/heavy-equipment-selling-price-prediction-challenge)
- [GitHub profile](https://github.com/amarkumar00)

## Reproducibility note

The public notebook contains code but no competition data or generated submission. Anyone with authorised dataset access can rerun the full pipeline. Reported metrics are labelled as results from the original executed notebook rather than results reproduced inside this cleaned copy.
