# Data contract

The dataset belongs to the IIT Madras/Kaggle Heavy Equipment Selling Price
Prediction challenge. Its access and redistribution rules are controlled by the
competition, so no raw or row-level derived data is committed to this repository.

## Expected local files

After joining the competition and downloading the authorised files, place them
here:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

The complete local pipeline requires `train.csv`. Run:

```bash
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
```

Generated files are written to `data/processed/`, `data/equipvalue.db`, and
`reports/`. These paths are ignored by Git because cleaned rows are still derived
from competition data.

## Minimum training columns

The cleaning pipeline requires:

- `TransactionID`
- `TransactionDate`
- `TargetValue`
- `ManufactureYear`
- `OperationalHoursMeter`

Optional columns such as `AssetID`, `Spec_BaseClass`, `RegionCode`, and
`UtilizationTier` are retained when available and power richer SQL and R
analysis.

If a required file or column is missing, the pipeline stops with an actionable
error instead of silently producing an incomplete report.
