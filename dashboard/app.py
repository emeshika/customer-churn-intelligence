"""
AI-Powered Customer Churn & Retention Intelligence Platform
Streamlit dashboard.

Expected SQLite tables (matching sql/schema.sql):
  customers(customer_id, gender, contract, payment_method,
            monthly_charges, tenure, actual_churn, ...)
  predictions(customer_id, churn_probability, risk_level, segment_name,
              predicted_at, ...)
  retention_actions(customer_id, suggested_action, ...)
Optional files:
  reports/model_metrics.json
  reports/figures/shap_summary_plot.png
  reports/figures/roc_curve.png
  reports/figures/pr_curve.png
  reports/customer_shap_values.csv
"""

from pathlib import Path
import json
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "churn_platform.db"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

st.set_page_config(
    page_title="Churn & Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.18);
        padding: 14px 16px;
        border-radius: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def table_exists(conn, table_name):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


@st.cache_data(ttl=60)
def load_data(db_path, cache_version=2):
    """Load tables and keep the latest prediction for each customer."""
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_file}")

    with sqlite3.connect(str(db_file)) as conn:
        if not table_exists(conn, "customers"):
            raise RuntimeError("Required table 'customers' was not found.")
        if not table_exists(conn, "predictions"):
            raise RuntimeError("Required table 'predictions' was not found.")

        customers = pd.read_sql_query("SELECT * FROM customers", conn)
        predictions = pd.read_sql_query("SELECT * FROM predictions", conn)

        if "customer_id" not in customers.columns:
            raise RuntimeError("The customers table must contain customer_id.")
        if "customer_id" not in predictions.columns:
            raise RuntimeError("The predictions table must contain customer_id.")

        # Keep the most recent prediction using the actual schema column.
        prediction_duplicate_count = int(predictions["customer_id"].duplicated().sum())
        if "predicted_at" in predictions.columns:
            predictions = predictions.sort_values(
                "predicted_at", na_position="first"
            )
        elif "prediction_timestamp" in predictions.columns:
            predictions = predictions.sort_values(
                "prediction_timestamp", na_position="first"
            )
        elif "prediction_id" in predictions.columns:
            predictions = predictions.sort_values("prediction_id")

        predictions = predictions.drop_duplicates("customer_id", keep="last")

        actions = pd.DataFrame(columns=["customer_id", "suggested_action"])
        if table_exists(conn, "retention_actions"):
            actions = pd.read_sql_query(
                "SELECT * FROM retention_actions", conn
            )

    customer_duplicate_count = int(customers["customer_id"].duplicated().sum())
    # The schema declares customer_id as a primary key. If a nonstandard
    # database contains duplicates, fail clearly rather than multiplying rows.
    if customer_duplicate_count:
        raise RuntimeError(
            "Duplicate customer_id values were found in customers. "
            "Resolve these records in the database before loading the dashboard."
        )

    df = customers.merge(
        predictions,
        on="customer_id",
        how="left",
        suffixes=("", "_prediction"),
        validate="one_to_one",
    )
    quality = {
        "customer_duplicates": customer_duplicate_count,
        "prediction_duplicates": prediction_duplicate_count,
    }
    return df, actions, quality


def fmt_pct(value):
    return "N/A" if pd.isna(value) else f"{value:.1f}%"


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def load_metrics():
    path = REPORTS_DIR / "model_metrics.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        st.warning("Could not read reports/model_metrics.json.")
        return None


def show_image_if_exists(path, caption):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Not available yet: {path.relative_to(ROOT_DIR)}")


st.sidebar.title("📊 Churn Intelligence")
st.sidebar.caption("Customer risk analysis and retention decision support")
page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Customer Risk & Retention",
        "Customer Segmentation",
        "Model Performance & XAI",
        "Retention Insights",
    ],
)
st.sidebar.divider()
st.sidebar.caption(
    "Predictions support human review; they do not guarantee churn or retention outcomes."
)

try:
    df, actions_df, data_quality = load_data(str(DB_PATH))
except Exception as exc:
    st.error(f"Unable to load dashboard data: {exc}")
    st.info(
        "Check that data/churn_platform.db exists and contains the required "
        "customers and predictions tables. The table definitions should match "
        "sql/schema.sql."
    )
    st.stop()

# Normalize values used in calculations and charts.
for col in ["churn_probability", "monthly_charges", "tenure",
            "actual_churn", "total_charges"]:
    if col in df.columns:
        df[col] = safe_numeric(df[col])

required_prediction_cols = {"churn_probability", "risk_level"}
missing_pred = required_prediction_cols - set(df.columns)
if missing_pred:
    st.error(
        "Missing prediction columns: "
        + ", ".join(sorted(missing_pred))
        + ". Check sql/schema.sql and the predictions table."
    )
    st.stop()

scored = df[df["churn_probability"].notna()].copy()
if scored.empty:
    st.warning(
        "No customer predictions are available yet. Populate the predictions "
        "table before using the prediction-related dashboard pages."
    )
    st.stop()

risk_order = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
risk_values = scored["risk_level"].dropna().astype(str).unique().tolist()
risk_labels = [risk for risk in risk_order if risk in risk_values]
risk_labels += sorted(r for r in risk_values if r not in risk_labels)

# ---------- Page 1 ----------
if page == "Executive Overview":
    st.title("Executive Overview")
    st.caption(
        "Portfolio-level churn, customer risk and estimated revenue exposure."
    )

    total_customers = int(df["customer_id"].nunique())
    scored_customers = int(scored["customer_id"].nunique())
    actual_churn_rate = None
    if "actual_churn" in scored and scored["actual_churn"].notna().any():
        actual_churn_rate = float(scored["actual_churn"].mean() * 100)

    at_risk = scored[
        scored["risk_level"].astype(str).str.contains(
            r"high|critical|very high", case=False, na=False
        )
    ]
    at_risk_count = int(at_risk["customer_id"].nunique())

    revenue_exposure = None
    if "monthly_charges" in at_risk.columns:
        revenue_exposure = at_risk["monthly_charges"].sum(min_count=1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{total_customers:,}")
    c2.metric("Customers with predictions", f"{scored_customers:,}")
    c3.metric(
        "Actual churn rate",
        fmt_pct(actual_churn_rate) if actual_churn_rate is not None else "N/A",
    )
    c4.metric("High / very-high risk", f"{at_risk_count:,}")

    if revenue_exposure is not None and not pd.isna(revenue_exposure):
        st.metric(
            "Monthly charges associated with at-risk customers",
            f"${revenue_exposure:,.2f}",
        )
        st.caption(
            "This is a risk-exposure proxy based on current monthly charges. "
            "It is not a prediction of revenue that will actually be lost."
        )

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Customer distribution by risk tier")
        risk_counts = (
            scored["risk_level"].fillna("Unknown")
            .value_counts()
            .rename_axis("Risk tier")
            .reset_index(name="Customers")
        )
        fig = px.pie(
            risk_counts,
            names="Risk tier",
            values="Customers",
            hole=0.55,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Average predicted churn probability by segment")
        if "segment_name" in scored.columns and scored["segment_name"].notna().any():
            seg = (
                scored.dropna(subset=["segment_name"])
                .groupby("segment_name", as_index=False)
                .agg(
                    avg_probability=("churn_probability", "mean"),
                    customers=("customer_id", "nunique"),
                )
            )
            seg["avg_probability_pct"] = seg["avg_probability"] * 100
            fig = px.bar(
                seg,
                x="segment_name",
                y="avg_probability_pct",
                text="avg_probability_pct",
                labels={
                    "segment_name": "Segment",
                    "avg_probability_pct": "Average predicted probability (%)",
                },
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False, yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Segment information is not available in the predictions table.")

    st.subheader("Portfolio data quality")
    q1, q2, q3 = st.columns(3)
    q1.metric(
        "Customers without a prediction",
        f"{int(df['churn_probability'].isna().sum()):,}",
    )
    q2.metric(
        "Missing risk labels",
        f"{int(scored['risk_level'].isna().sum()):,}",
    )
    q3.metric(
        "Duplicate prediction rows handled",
        f"{data_quality['prediction_duplicates']:,}",
    )

# ---------- Page 2 ----------
elif page == "Customer Risk & Retention":
    st.title("Customer Risk & Retention")
    st.caption("Review individual predictions, customer attributes and suggested actions.")

    f1, f2, f3 = st.columns(3)
    risk_filter = f1.multiselect(
        "Risk tier", options=risk_labels, default=risk_labels
    )
    segment_options = (
        sorted(scored["segment_name"].dropna().astype(str).unique().tolist())
        if "segment_name" in scored.columns else []
    )
    segment_filter = f2.multiselect(
        "Customer segment", options=segment_options, default=segment_options
    )
    min_prob = f3.slider(
        "Minimum churn probability", 0.0, 1.0, 0.0, 0.05
    )

    filtered = scored[scored["risk_level"].astype(str).isin(risk_filter)]
    if "segment_name" in filtered.columns and segment_options:
        filtered = filtered[
            filtered["segment_name"].astype(str).isin(segment_filter)
        ]
    filtered = filtered[filtered["churn_probability"] >= min_prob]

    id_options = filtered["customer_id"].astype(str).sort_values().tolist()
    if not id_options:
        st.warning("No customers match the selected filters.")
    else:
        selected_id = st.selectbox("Select customer", id_options)
        customer = filtered[
            filtered["customer_id"].astype(str) == selected_id
        ].iloc[0]

        a, b, c, d = st.columns(4)
        a.metric("Customer ID", selected_id)
        b.metric("Churn probability", f"{customer['churn_probability']:.1%}")
        c.metric("Risk tier", str(customer["risk_level"]))
        d.metric("Segment", str(customer.get("segment_name", "N/A")))

        st.subheader("Customer attributes")
        display_cols = [
            col for col in [
                "contract", "payment_method", "tenure", "monthly_charges",
                "gender", "total_charges",
            ] if col in customer.index
        ]
        if display_cols:
            st.dataframe(
                pd.DataFrame([customer[display_cols].to_dict()]),
                hide_index=True,
                use_container_width=True,
            )

        st.subheader("Suggested retention actions")
        if (
            not actions_df.empty
            and {"customer_id", "suggested_action"}.issubset(actions_df.columns)
        ):
            cust_actions = actions_df[
                actions_df["customer_id"].astype(str) == selected_id
            ]["suggested_action"].dropna().tolist()
            if cust_actions:
                for action in cust_actions:
                    st.info(action)
            else:
                st.caption("No retention action is recorded for this customer.")
        else:
            st.caption("Retention-action data is not available.")

        st.subheader("Prediction explanation")
        st.caption(
            "Individual SHAP explanations require saved per-customer SHAP values. "
            "A global SHAP plot cannot explain this specific customer."
        )
        shap_csv = REPORTS_DIR / "customer_shap_values.csv"
        if shap_csv.exists():
            shap_df = pd.read_csv(shap_csv)
            if "customer_id" in shap_df.columns:
                row = shap_df[
                    shap_df["customer_id"].astype(str) == selected_id
                ]
                if not row.empty:
                    vals = row.drop(columns=["customer_id"]).iloc[0]
                    vals = pd.to_numeric(vals, errors="coerce").dropna()
                    vals = vals.reindex(
                        vals.abs().sort_values(ascending=False).head(10).index
                    )
                    fig = px.bar(
                        x=vals.values,
                        y=vals.index,
                        orientation="h",
                        labels={"x": "SHAP value", "y": "Feature"},
                        title="Top individual feature contributions",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No saved SHAP explanation found for this customer.")
            else:
                st.warning("customer_shap_values.csv must include customer_id.")
        else:
            st.info("Individual SHAP values have not been exported yet.")

        st.subheader("Filtered customer list")
        list_cols = ["customer_id", "churn_probability", "risk_level"]
        if "segment_name" in filtered.columns:
            list_cols.append("segment_name")
        st.dataframe(
            filtered[list_cols].sort_values(
                "churn_probability", ascending=False
            ),
            hide_index=True,
            use_container_width=True,
        )

# ---------- Page 3 ----------
elif page == "Customer Segmentation":
    st.title("Customer Segmentation")
    st.caption("Explore the observed characteristics of the clustering output.")

    if "segment_name" not in scored.columns:
        st.warning("segment_name is not available in the predictions table.")
        st.stop()

    seg = scored.dropna(subset=["segment_name"]).copy()
    if seg.empty:
        st.warning("No assigned customer segments are available.")
        st.stop()

    group_cols = {"Customers": ("customer_id", "nunique")}
    if "tenure" in seg.columns:
        group_cols["Average tenure (months)"] = ("tenure", "mean")
    if "monthly_charges" in seg.columns:
        group_cols["Average monthly charges"] = ("monthly_charges", "mean")
    group_cols["Average predicted churn probability"] = (
        "churn_probability", "mean"
    )
    if "actual_churn" in seg.columns and seg["actual_churn"].notna().any():
        group_cols["Actual churn rate"] = ("actual_churn", "mean")

    profile = seg.groupby("segment_name").agg(**group_cols).reset_index()
    profile["Average predicted churn probability"] *= 100
    if "Actual churn rate" in profile.columns:
        profile["Actual churn rate"] *= 100

    st.subheader("Segment profiles")
    st.dataframe(profile, hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    with left:
        if {"tenure", "monthly_charges"}.issubset(seg.columns):
            st.subheader("Segment landscape")
            hover_cols = ["customer_id", "risk_level"]
            fig = px.scatter(
                seg,
                x="tenure",
                y="monthly_charges",
                color="segment_name",
                hover_data=[col for col in hover_cols if col in seg.columns],
                labels={
                    "tenure": "Tenure (months)",
                    "monthly_charges": "Monthly charges",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tenure or monthly_charges is unavailable.")

    with right:
        st.subheader("Segment size")
        sizes = (
            seg.groupby("segment_name", as_index=False)["customer_id"]
            .nunique()
        )
        sizes.columns = ["Segment", "Customers"]
        fig = px.bar(sizes, x="Segment", y="Customers", text="Customers")
        st.plotly_chart(fig, use_container_width=True)

    if "actual_churn" in seg.columns and seg["actual_churn"].notna().any():
        st.subheader("Actual churn rate by segment")
        churn_seg = (
            seg.groupby("segment_name", as_index=False)["actual_churn"].mean()
        )
        churn_seg["actual_churn_pct"] = churn_seg["actual_churn"] * 100
        fig = px.bar(
            churn_seg,
            x="segment_name",
            y="actual_churn_pct",
            labels={
                "segment_name": "Segment",
                "actual_churn_pct": "Actual churn rate (%)",
            },
            text="actual_churn_pct",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Cluster labels are identifiers, not inherently meaningful personas. "
        "Interpret names only after reviewing the underlying feature profiles."
    )

# ---------- Page 4 ----------
elif page == "Model Performance & XAI":
    st.title("Model Performance & Explainable AI")
    st.caption(
        "Metrics should come from the held-out evaluation run, not manually entered values."
    )

    metrics = load_metrics()
    if metrics is None:
        st.warning(
            "No reports/model_metrics.json found. Export the actual test-set "
            "metrics from the model evaluation pipeline to display them here."
        )
    elif not isinstance(metrics, dict) or not metrics:
        st.warning("model_metrics.json must contain a non-empty JSON object.")
    else:
        st.subheader("Held-out test performance")
        model_names = list(metrics.keys())
        selected_model = st.selectbox("Model", model_names)
        model_metrics = metrics[selected_model]
        if isinstance(model_metrics, dict):
            numeric_metrics = {
                name: value for name, value in model_metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if numeric_metrics:
                metric_cols = st.columns(min(6, len(numeric_metrics)))
                for i, (name, value) in enumerate(numeric_metrics.items()):
                    metric_cols[i % len(metric_cols)].metric(
                        name.replace("_", " ").title(), f"{value:.4f}"
                    )
            else:
                st.info("No numeric metrics are available for this model.")

        comparison = []
        for model_name, values in metrics.items():
            if isinstance(values, dict):
                row = {"Model": model_name}
                row.update({
                    key: value for key, value in values.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                })
                comparison.append(row)
        if comparison:
            st.subheader("Model comparison")
            st.dataframe(
                pd.DataFrame(comparison),
                hide_index=True,
                use_container_width=True,
            )

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("ROC curve")
        show_image_if_exists(
            FIGURES_DIR / "roc_curve.png",
            "ROC curve on the held-out test set",
        )
    with right:
        st.subheader("Precision–Recall curve")
        show_image_if_exists(
            FIGURES_DIR / "pr_curve.png",
            "Precision–Recall curve on the held-out test set",
        )

    st.subheader("Global SHAP explanation")
    show_image_if_exists(
        FIGURES_DIR / "shap_summary_plot.png",
        "Global SHAP summary plot generated from the selected model",
    )
    st.caption(
        "Report the test split, positive class, threshold, and metric definitions "
        "alongside results. Compare precision/recall in light of the business cost "
        "of false positives and false negatives."
    )

# ---------- Page 5 ----------
elif page == "Retention Insights":
    st.title("Retention Insights")
    st.caption(
        "Prioritize review of customers based on model risk and available business attributes."
    )

    high = scored[
        scored["risk_level"].astype(str).str.contains(
            r"high|critical|very high", case=False, na=False
        )
    ].copy()
    if high.empty:
        st.info("No customers are currently labeled High Risk or Very High Risk.")
    else:
        sort_cols = ["churn_probability"]
        ascending = [False]
        if "monthly_charges" in high.columns:
            sort_cols.append("monthly_charges")
            ascending.append(False)
        high = high.sort_values(sort_cols, ascending=ascending)

        st.metric(
            "Customers flagged for review",
            f"{high['customer_id'].nunique():,}",
        )
        st.caption(
            "This list is a decision-support queue. Risk labels do not prove that "
            "a customer will churn, and suggested actions should be reviewed by staff."
        )

        columns = ["customer_id", "churn_probability", "risk_level"]
        for col in ["segment_name", "contract", "tenure", "monthly_charges"]:
            if col in high.columns:
                columns.append(col)
        st.dataframe(
            high[columns],
            hide_index=True,
            use_container_width=True,
        )

        if (
            not actions_df.empty
            and {"customer_id", "suggested_action"}.issubset(actions_df.columns)
        ):
            st.subheader("Recorded suggested actions")
            action_summary = (
                actions_df.groupby("suggested_action", as_index=False)
                .size()
                .rename(columns={"size": "Customers"})
                .sort_values("Customers", ascending=False)
            )
            st.dataframe(
                action_summary,
                hide_index=True,
                use_container_width=True,
            )

st.sidebar.divider()
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()
