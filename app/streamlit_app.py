"""Streamlit interface for single and chunked heavy-equipment predictions."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.batch_predict import predict_dataframe_in_chunks
from src.predictor import PredictionInputError, PricePredictor

st.set_page_config(
    page_title="EquipValue Price Intelligence",
    page_icon="🏗️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetric"] {border: 1px solid #dbe5e1; border-radius: 12px; padding: 14px;}
    .decision-note {background:#f4f8f6; border-left:4px solid #2b6e59; padding:14px; border-radius:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> PricePredictor:
    return PricePredictor.load(PROJECT_ROOT / "models")


def option_values(context: dict, column: str, fallback: list[str]) -> list[str]:
    values = context.get("form_options", {}).get(column, [])
    return [str(value) for value in values] or fallback


def default_text(context: dict, column: str, fallback: str = "") -> str:
    values = context.get("form_options", {}).get(column, [])
    return str(values[0]) if values else fallback


st.title("EquipValue Price Intelligence")
st.caption(
    "Explainable heavy-equipment selling-price estimates with time-aware model evidence"
)

try:
    predictor = load_predictor()
except FileNotFoundError as error:
    st.error(str(error))
    st.code(
        "python -m src.train_classical_models\n"
        "python -m src.build_business_context\n"
        "streamlit run app/streamlit_app.py"
    )
    st.stop()

context = predictor.business_context
training_min = pd.Timestamp(context["training_date_min"])
training_max = pd.Timestamp(context["training_date_max"])

st.markdown(
    f"""
    <div class="decision-note">
    <strong>Decision-support boundary:</strong> {context["decision_boundary"]}<br>
    Training period: {training_min.date()} to {training_max.date()}.
    </div>
    """,
    unsafe_allow_html=True,
)

single_tab, batch_tab, model_tab = st.tabs(
    ["Single equipment", "Batch CSV", "Model card"]
)

with single_tab:
    st.subheader("Estimate one equipment selling price")
    st.write(
        "Use known auction and equipment details. Blank optional fields reduce confidence."
    )

    with st.form("single_prediction_form"):
        left, middle, right = st.columns(3)
        with left:
            transaction_date = st.date_input(
                "Expected sale date",
                value=training_max.date(),
                help="Dates outside the historical training period trigger a warning.",
            )
            manufacture_year = st.number_input(
                "Manufacture year",
                min_value=1940,
                max_value=transaction_date.year + 1,
                value=min(2005, transaction_date.year),
            )
            hours_unknown = st.checkbox("Operational hours unknown")
            operational_hours = st.number_input(
                "Operational hours",
                min_value=0.0,
                value=4000.0,
                disabled=hours_unknown,
            )
            utilization = st.selectbox(
                "Utilization tier",
                option_values(context, "UtilizationTier", ["Low", "Medium", "High"]),
            )
            scale = st.selectbox(
                "Equipment scale",
                option_values(
                    context, "AssetScaleFactor", ["Small", "Medium", "Large"]
                ),
            )

        with middle:
            inventory_category = st.selectbox(
                "Inventory category",
                option_values(context, "InventoryGroupCategory", ["TEX", "MG", "WL"]),
            )
            region = st.selectbox(
                "Region",
                option_values(
                    context, "RegionCode", ["Florida", "Texas", "California"]
                ),
            )
            cabin = st.selectbox(
                "Cabin type",
                option_values(context, "CabinType", ["EROPS", "OROPS"]),
            )
            forks = st.selectbox(
                "Forks",
                [""] + option_values(context, "Forks", ["None or Unspecified", "Yes"]),
            )
            drivetrain = st.selectbox(
                "Drivetrain",
                [""]
                + option_values(
                    context, "DrivetrainType", ["Standard", "Powershift", "Hydrostatic"]
                ),
            )

        with right:
            product_config = st.text_input(
                "Product configuration ID",
                value=default_text(context, "ProductConfigID", "unknown"),
            )
            base_class = st.text_input(
                "Base class", value=default_text(context, "Spec_BaseClass", "unknown")
            )
            sub_class = st.text_input(
                "Sub-class", value=default_text(context, "Spec_SubClass", "unknown")
            )
            functional_class = st.text_input(
                "Functional classification",
                value=default_text(context, "FunctionalClassification", "unknown"),
            )
            description = st.text_input(
                "Full equipment descriptor",
                value=default_text(context, "Spec_FullDescriptor", "unknown"),
            )

        with st.expander("Additional model fields"):
            data_origin = st.selectbox(
                "Data origin code",
                option_values(context, "DataOriginCode", ["ADI149"]),
            )
            vendor = st.selectbox(
                "Vendor partner ID",
                option_values(context, "VendorPartnerID", ["UJF"]),
            )
            asset_id = st.text_input("Asset ID (optional)", value="new-equipment")

        submitted = st.form_submit_button("Estimate selling price", type="primary")

    if submitted:
        row = pd.DataFrame(
            [
                {
                    "TransactionID": "interactive-prediction",
                    "AssetID": asset_id or pd.NA,
                    "TransactionDate": transaction_date.isoformat(),
                    "ManufactureYear": manufacture_year,
                    "OperationalHoursMeter": pd.NA
                    if hours_unknown
                    else operational_hours,
                    "UtilizationTier": utilization,
                    "AssetScaleFactor": scale,
                    "Spec_FullDescriptor": description or pd.NA,
                    "ProductConfigID": product_config or pd.NA,
                    "Spec_BaseClass": base_class or pd.NA,
                    "Spec_SubClass": sub_class or pd.NA,
                    "FunctionalClassification": functional_class or pd.NA,
                    "RegionCode": region,
                    "InventoryGroupCategory": inventory_category,
                    "CabinType": cabin,
                    "Forks": forks or pd.NA,
                    "DrivetrainType": drivetrain or pd.NA,
                    "DataOriginCode": data_origin,
                    "VendorPartnerID": vendor,
                }
            ]
        )
        try:
            result = predictor.predict(row).iloc[0]
        except PredictionInputError as error:
            st.error(str(error))
        else:
            estimate, price_range, comparison = st.columns(3)
            estimate.metric(
                "Estimated selling price", f"${result.estimated_selling_price:,.0f}"
            )
            price_range.metric(
                "Empirical model range",
                f"${result.range_lower:,.0f} – ${result.range_upper:,.0f}",
            )
            comparison.metric(
                "Similar-category historical average",
                f"${result.similar_category_historical_average:,.0f}",
            )
            st.markdown("#### Major model influences")
            for item in result.major_influencing_features.split("; "):
                st.write(f"- {item}")
            if result.unfamiliar_input_warning != "No major unfamiliar inputs detected":
                st.warning(result.unfamiliar_input_warning)
            else:
                st.success(result.unfamiliar_input_warning)
            st.info(result.uncertainty_note)
            st.error(result.human_inspection_required)

with batch_tab:
    st.subheader("Chunked CSV prediction")
    st.write(
        "Upload a CSV containing at least `TransactionDate`. Manufacture year and "
        "equipment details improve usefulness. Invalid rows are separated instead of "
        "silently dropped."
    )
    template_path = PROJECT_ROOT / "examples" / "batch_input_template.csv"
    if template_path.exists():
        st.download_button(
            "Download CSV template",
            data=template_path.read_text(encoding="utf-8"),
            file_name="equipvalue_batch_template.csv",
            mime="text/csv",
        )
    uploaded = st.file_uploader("Equipment CSV", type=["csv"])
    chunk_size = st.select_slider(
        "Rows processed per chunk", options=[250, 500, 1_000, 2_500, 5_000], value=1_000
    )
    if uploaded is not None:
        try:
            uploaded_data = pd.read_csv(uploaded, low_memory=False)
        except (
            pd.errors.ParserError,
            UnicodeDecodeError,
            ValueError,
            TypeError,
        ) as error:
            st.error(f"Could not read CSV: {error}")
        else:
            st.write(
                f"Loaded {len(uploaded_data):,} rows and {len(uploaded_data.columns)} columns."
            )
            if st.button("Run batch prediction", type="primary"):
                predictions, invalid = predict_dataframe_in_chunks(
                    uploaded_data, predictor, chunk_size=chunk_size
                )
                one, two, three = st.columns(3)
                one.metric("Input rows", f"{len(uploaded_data):,}")
                two.metric("Predicted rows", f"{len(predictions):,}")
                three.metric("Invalid rows", f"{len(invalid):,}")
                if not predictions.empty:
                    st.dataframe(predictions.head(100), use_container_width=True)
                    output = io.StringIO()
                    predictions.to_csv(output, index=False)
                    st.download_button(
                        "Download predictions",
                        data=output.getvalue(),
                        file_name="equipvalue_predictions.csv",
                        mime="text/csv",
                    )
                if not invalid.empty:
                    st.warning("Some rows need correction before scoring.")
                    st.dataframe(invalid.head(100), use_container_width=True)
                    invalid_output = io.StringIO()
                    invalid.to_csv(invalid_output, index=False)
                    st.download_button(
                        "Download invalid-row report",
                        data=invalid_output.getvalue(),
                        file_name="equipvalue_invalid_rows.csv",
                        mime="text/csv",
                    )

with model_tab:
    st.subheader("Model card and limitations")
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Time-aware LightGBM RMSLE", "0.3493")
    metric_two.metric("Training rows", f"{context['training_rows']:,}")
    metric_three.metric(
        "OOF empirical range coverage",
        f"{context['uncertainty']['empirical_coverage']:.0%}",
    )
    st.markdown(
        """
        **What the app can do**

        - Provide a consistent starting estimate for reserve-price and budget planning.
        - Compare an estimate with historical category averages.
        - Highlight model features that raised or lowered the log-price prediction.
        - Flag unfamiliar categories and dates outside the training period.

        **What the app cannot do**

        - Inspect mechanical condition, title history, attachments, or hidden damage.
        - Guarantee a sale price or a statistically calibrated confidence interval.
        - Replace an appraiser, mechanic, auction specialist, or commercial judgement.
        - Prove that an input feature caused the predicted price.
        """
    )
    st.caption(
        "This local application is tested software, but it is not described as a "
        "public deployment until a live URL is published and independently checked."
    )
