from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Customer Review Sentiment Analysis System",
    layout="wide",
    initial_sidebar_state="expanded",
)


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"

PIPELINE_PATH = MODELS_DIR / "best_sentiment_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
SCORED_DATA_PATH = PROCESSED_DIR / "reviews_with_predictions.csv"
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_reviews.csv"
RAW_DATA_PATH = RAW_DIR / "Womens Clothing E-Commerce Reviews.csv"

DEFAULT_REVIEW_COLUMNS = [
    "clean_review",
    "full_review",
    "review_text",
    "Review Text",
    "review",
    "text",
    "comment",
    "feedback",
    "review_body",
]
DEFAULT_TITLE_COLUMNS = ["title", "Title", "headline", "summary"]


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_pipeline():
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {PIPELINE_PATH}. Run the training notebook first."
        )
    return joblib.load(PIPELINE_PATH)


@st.cache_data
def load_metadata() -> dict:
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_local_dataset() -> Optional[pd.DataFrame]:
    if SCORED_DATA_PATH.exists():
        return pd.read_csv(SCORED_DATA_PATH)
    if CLEANED_DATA_PATH.exists():
        df = pd.read_csv(CLEANED_DATA_PATH)
        return score_dataframe(df.copy(), text_column=detect_preferred_text_column(df))
    if RAW_DATA_PATH.exists():
        df = pd.read_csv(RAW_DATA_PATH)
        df = standardize_raw_dataframe(df)
        df = prepare_base_dataframe(df)
        return score_dataframe(df.copy(), text_column="clean_review")
    return None


def standardize_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Clothing ID": "clothing_id",
        "Age": "age",
        "Title": "title",
        "Review Text": "review_text",
        "Rating": "rating",
        "Recommended IND": "recommended_ind",
        "Positive Feedback Count": "positive_feedback_count",
        "Division Name": "division_name",
        "Department Name": "department_name",
        "Class Name": "class_name",
    }
    return df.rename(columns=rename_map)


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "review_text" not in df.columns and "Review Text" in df.columns:
        df["review_text"] = df["Review Text"]
    if "title" not in df.columns and "Title" in df.columns:
        df["title"] = df["Title"]

    if "review_text" in df.columns:
        df = df.dropna(subset=["review_text"]).copy()
        df["review_text"] = df["review_text"].astype(str).str.strip()
        df = df[df["review_text"] != ""].copy()

    if "title" not in df.columns:
        df["title"] = ""
    df["title"] = df["title"].fillna("").astype(str).str.strip()

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

        def map_sentiment(rating):
            if rating in [1, 2]:
                return "negative"
            if rating == 3:
                return "neutral"
            if rating in [4, 5]:
                return "positive"
            return np.nan

        if "sentiment" not in df.columns:
            df["sentiment"] = df["rating"].apply(map_sentiment)

    df["full_review"] = (
        df.get("title", "").fillna("").astype(str)
        + " "
        + df.get("review_text", "").fillna("").astype(str)
    ).str.strip()
    df["clean_review"] = df["full_review"].apply(clean_text)
    return df


def detect_preferred_text_column(df: pd.DataFrame) -> str:
    for col in DEFAULT_REVIEW_COLUMNS:
        if col in df.columns:
            return col
    object_cols = df.select_dtypes(include="object").columns.tolist()
    if not object_cols:
        raise ValueError("No text-like column was found in the dataset.")
    return object_cols[0]


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def get_prediction_details(texts: Iterable[str], pipeline) -> tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    texts = pd.Series(list(texts)).fillna("").astype(str)
    preds = pipeline.predict(texts)

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(texts)
        class_labels = list(pipeline.classes_)
        conf = probs.max(axis=1)
        prob_df = pd.DataFrame(probs, columns=[f"score_{c}" for c in class_labels])
        return preds, conf, prob_df

    if hasattr(pipeline, "decision_function"):
        margins = pipeline.decision_function(texts)
        class_labels = list(pipeline.classes_)

        if np.ndim(margins) == 1:
            margins = np.column_stack([-margins, margins])
            if len(class_labels) != 2:
                class_labels = [f"class_{i}" for i in range(margins.shape[1])]

        probs = softmax(np.asarray(margins))
        conf = probs.max(axis=1)
        prob_df = pd.DataFrame(probs, columns=[f"score_{c}" for c in class_labels])
        return preds, conf, prob_df

    conf = np.full(shape=len(texts), fill_value=np.nan)
    return preds, conf, None


def score_dataframe(
    df: pd.DataFrame,
    text_column: str,
    title_column: Optional[str] = None,
) -> pd.DataFrame:
    pipeline = load_pipeline()
    scored = df.copy()

    if title_column and title_column in scored.columns and text_column in scored.columns:
        combined_text = (
            scored[title_column].fillna("").astype(str)
            + " "
            + scored[text_column].fillna("").astype(str)
        ).str.strip()
    else:
        combined_text = scored[text_column].fillna("").astype(str)

    scored["app_input_text"] = combined_text
    scored["app_clean_text"] = scored["app_input_text"].apply(clean_text)

    preds, conf, score_df = get_prediction_details(scored["app_clean_text"], pipeline)
    scored["predicted_sentiment"] = preds
    scored["prediction_confidence"] = conf

    if score_df is not None:
        score_df.index = scored.index
        scored = pd.concat([scored, score_df], axis=1)

    return scored


def make_downloadable_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def plot_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, rotation: int = 0):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    st.pyplot(fig)


def plot_stacked_bar(df: pd.DataFrame, title: str, xlabel: str, ylabel: str, rotation: int = 0):
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    ax.legend(title="Sentiment")
    st.pyplot(fig)


def age_bucket(series: pd.Series) -> pd.Series:
    bins = [0, 24, 34, 44, 54, 64, 120]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


pipeline = None
pipeline_error = None
try:
    pipeline = load_pipeline()
except Exception as exc:  # pragma: no cover
    pipeline_error = str(exc)

metadata = load_metadata()
local_df = load_local_dataset()


st.title("Customer Review Sentiment Analysis and Feedback Intelligence System")
st.caption(
    "Classify review sentiment, analyse uploaded review files, and explore customer feedback patterns across the fashion e-commerce dataset."
)

with st.sidebar:
    st.subheader("Project assets")
    if pipeline_error:
        st.error("Model not available")
        st.caption(pipeline_error)
    else:
        st.success("Model loaded")
        model_name = metadata.get("best_model_name", type(pipeline).__name__)
        st.caption(f"Best model: {model_name}")

    st.write("Expected paths")
    st.code(
        "models/best_sentiment_pipeline.joblib\n"
        "models/model_metadata.json\n"
        "data/processed/cleaned_reviews.csv\n"
        "data/processed/reviews_with_predictions.csv",
        language="text",
    )

    st.subheader("How the app scores text")
    st.caption(
        "The app combines title and review text where available, applies the same cleaning logic used in the project pipeline, then predicts positive, neutral, or negative sentiment."
    )

if pipeline_error:
    st.stop()


dashboard_tab, single_tab, batch_tab, explorer_tab = st.tabs(
    [
        "Dashboard",
        "Single Review Prediction",
        "Batch CSV Analysis",
        "Data Explorer",
    ]
)

with dashboard_tab:
    st.subheader("Customer feedback dashboard")

    if local_df is None or local_df.empty:
        st.info("No processed project dataset was found yet. Add the cleaned or scored CSV files to the expected folders to populate the dashboard.")
    else:
        filtered = local_df.copy()

        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            if "department_name" in filtered.columns:
                department_options = sorted(filtered["department_name"].dropna().astype(str).unique())
                selected_departments = st.multiselect("Department", department_options)
                if selected_departments:
                    filtered = filtered[filtered["department_name"].astype(str).isin(selected_departments)]

        with filter_col2:
            if "class_name" in filtered.columns:
                class_options = sorted(filtered["class_name"].dropna().astype(str).unique())
                selected_classes = st.multiselect("Class", class_options)
                if selected_classes:
                    filtered = filtered[filtered["class_name"].astype(str).isin(selected_classes)]

        with filter_col3:
            sentiment_source = "predicted_sentiment" if "predicted_sentiment" in filtered.columns else "sentiment"
            if sentiment_source in filtered.columns:
                sentiment_options = sorted(filtered[sentiment_source].dropna().astype(str).unique())
                selected_sentiments = st.multiselect("Sentiment", sentiment_options)
                if selected_sentiments:
                    filtered = filtered[filtered[sentiment_source].astype(str).isin(selected_sentiments)]

        with filter_col4:
            if "age" in filtered.columns and filtered["age"].notna().any():
                min_age = int(filtered["age"].min())
                max_age = int(filtered["age"].max())
                age_range = st.slider("Age range", min_age, max_age, (min_age, max_age))
                filtered = filtered[filtered["age"].between(age_range[0], age_range[1])]

        metric_cols = st.columns(4)
        metric_cols[0].metric("Reviews", f"{len(filtered):,}")

        if "rating" in filtered.columns:
            metric_cols[1].metric("Average rating", f"{filtered['rating'].mean():.2f}")
        else:
            metric_cols[1].metric("Average rating", "N/A")

        sentiment_source = "predicted_sentiment" if "predicted_sentiment" in filtered.columns else "sentiment"
        if sentiment_source in filtered.columns:
            positive_share = (filtered[sentiment_source].eq("positive").mean() * 100) if len(filtered) else 0
            metric_cols[2].metric("Positive share", f"{positive_share:.1f}%")
        else:
            metric_cols[2].metric("Positive share", "N/A")

        if "recommended_ind" in filtered.columns:
            rec_share = filtered["recommended_ind"].fillna(0).astype(float).mean() * 100
            metric_cols[3].metric("Recommendation rate", f"{rec_share:.1f}%")
        else:
            metric_cols[3].metric("Recommendation rate", "N/A")

        st.markdown("### Insights")
        st.markdown("#### Data-driven insights")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if sentiment_source in filtered.columns:
                sentiment_counts = filtered[sentiment_source].value_counts()
                plot_bar(sentiment_counts, "Sentiment distribution", "Sentiment", "Review count")

        with chart_col2:
            if "rating" in filtered.columns:
                rating_counts = filtered["rating"].value_counts().sort_index()
                plot_bar(rating_counts, "Rating distribution", "Rating", "Review count")

        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            if {"department_name", sentiment_source}.issubset(filtered.columns):
                dept_sentiment = pd.crosstab(filtered["department_name"], filtered[sentiment_source])
                dept_sentiment = dept_sentiment.loc[dept_sentiment.sum(axis=1).sort_values(ascending=False).head(10).index]
                if not dept_sentiment.empty:
                    plot_stacked_bar(
                        dept_sentiment,
                        "Top departments by sentiment",
                        "Department",
                        "Review count",
                        rotation=35,
                    )

        with chart_col4:
            if "age" in filtered.columns and sentiment_source in filtered.columns:
                age_df = filtered.copy()
                age_df["age_group"] = age_bucket(pd.to_numeric(age_df["age"], errors="coerce"))
                age_sentiment = pd.crosstab(age_df["age_group"], age_df[sentiment_source])
                if not age_sentiment.empty:
                    plot_stacked_bar(
                        age_sentiment,
                        "Sentiment by age group",
                        "Age group",
                        "Review count",
                    )

        st.markdown("#### Model-driven insights")
        model_col1, model_col2 = st.columns(2)

        with model_col1:
            if "prediction_confidence" in filtered.columns and filtered["prediction_confidence"].notna().any():
                avg_conf = filtered["prediction_confidence"].mean()
                low_conf = (filtered["prediction_confidence"] < 0.60).sum()
                st.metric("Average confidence", f"{avg_conf:.3f}")
                st.metric("Low-confidence reviews", f"{low_conf:,}")
            else:
                st.info("Confidence scores are not available for the current model type.")

        with model_col2:
            if {"sentiment", "predicted_sentiment"}.issubset(filtered.columns):
                agreement = (filtered["sentiment"] == filtered["predicted_sentiment"]).mean() * 100
                st.metric("Label agreement on loaded dataset", f"{agreement:.1f}%")
                st.caption("This is a descriptive agreement view on the loaded data, not an unseen test-set metric.")

        if "prediction_confidence" in filtered.columns and filtered["prediction_confidence"].notna().any():
            st.markdown("Lowest-confidence reviews")
            low_conf_df = filtered.sort_values("prediction_confidence", ascending=True)
            cols_to_show = [
                c for c in [
                    "title",
                    "review_text",
                    "sentiment",
                    "predicted_sentiment",
                    "prediction_confidence",
                    "department_name",
                    "class_name",
                    "rating",
                ] if c in low_conf_df.columns
            ]
            st.dataframe(low_conf_df[cols_to_show].head(20), use_container_width=True)

with single_tab:
    st.subheader("Predict sentiment for a single review")

    single_title = st.text_input("Optional review title")
    single_review = st.text_area("Review text", height=180)

    if st.button("Predict sentiment", type="primary"):
        if not single_review.strip():
            st.warning("Enter review text to generate a prediction.")
        else:
            input_text = f"{single_title} {single_review}".strip()
            clean_input = clean_text(input_text)
            pred, conf, score_df = get_prediction_details([clean_input], pipeline)

            result_cols = st.columns(2)
            result_cols[0].metric("Predicted sentiment", str(pred[0]).title())
            if not np.isnan(conf[0]):
                result_cols[1].metric("Confidence", f"{conf[0]:.3f}")
            else:
                result_cols[1].metric("Confidence", "N/A")

            if score_df is not None and not score_df.empty:
                display_scores = score_df.T.reset_index()
                display_scores.columns = ["Class", "Score"]
                display_scores["Class"] = display_scores["Class"].str.replace("score_", "", regex=False)
                st.dataframe(display_scores, use_container_width=True)

            st.markdown("Processed input")
            st.code(clean_input, language="text")

with batch_tab:
    st.subheader("Analyse a CSV of reviews")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview")
        st.dataframe(batch_df.head(), use_container_width=True)

        detected_review_col = next((c for c in DEFAULT_REVIEW_COLUMNS if c in batch_df.columns), None)
        detected_title_col = next((c for c in DEFAULT_TITLE_COLUMNS if c in batch_df.columns), None)

        selectable_cols = batch_df.columns.tolist()
        review_col = st.selectbox(
            "Review text column",
            selectable_cols,
            index=selectable_cols.index(detected_review_col) if detected_review_col in selectable_cols else 0,
        )

        title_options = ["None"] + selectable_cols
        title_default = title_options.index(detected_title_col) if detected_title_col in selectable_cols else 0
        title_col = st.selectbox("Optional title column", title_options, index=title_default)
        title_col = None if title_col == "None" else title_col

        if st.button("Run batch analysis", type="primary"):
            scored_batch = score_dataframe(batch_df, text_column=review_col, title_column=title_col)
            st.success("Batch analysis complete")

            batch_metrics = st.columns(3)
            batch_metrics[0].metric("Rows analysed", f"{len(scored_batch):,}")
            batch_metrics[1].metric(
                "Positive share",
                f"{(scored_batch['predicted_sentiment'].eq('positive').mean() * 100):.1f}%",
            )
            if scored_batch["prediction_confidence"].notna().any():
                batch_metrics[2].metric(
                    "Average confidence",
                    f"{scored_batch['prediction_confidence'].mean():.3f}",
                )
            else:
                batch_metrics[2].metric("Average confidence", "N/A")

            st.dataframe(scored_batch.head(25), use_container_width=True)

            download_bytes = make_downloadable_csv(scored_batch)
            st.download_button(
                label="Download scored CSV",
                data=download_bytes,
                file_name="reviews_scored.csv",
                mime="text/csv",
            )

with explorer_tab:
    st.subheader("Explore the local project dataset")

    if local_df is None or local_df.empty:
        st.info("No local project dataset is available yet.")
    else:
        explorer_df = local_df.copy()
        search_term = st.text_input("Search review text")

        text_source = None
        for candidate in ["review_text", "full_review", "clean_review", "app_input_text"]:
            if candidate in explorer_df.columns:
                text_source = candidate
                break

        if search_term and text_source:
            explorer_df = explorer_df[
                explorer_df[text_source].fillna("").astype(str).str.contains(search_term, case=False, regex=False)
            ]

        columns_to_show = [
            c for c in [
                "title",
                "review_text",
                "rating",
                "sentiment",
                "predicted_sentiment",
                "prediction_confidence",
                "department_name",
                "class_name",
                "age",
                "recommended_ind",
                "positive_feedback_count",
            ] if c in explorer_df.columns
        ]
        st.dataframe(explorer_df[columns_to_show], use_container_width=True, height=500)

        st.download_button(
            label="Download local dataset view",
            data=make_downloadable_csv(explorer_df[columns_to_show]),
            file_name="project_reviews_view.csv",
            mime="text/csv",
        )
