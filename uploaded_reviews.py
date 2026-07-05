from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


XQUIK_REVIEW_COLUMNS = [
    "full_text",
    "tweet_text",
    "message",
    "content",
    "body",
]


def review_column_candidates(base_columns: Iterable[str]) -> list[str]:
    return list(dict.fromkeys([*base_columns, *XQUIK_REVIEW_COLUMNS]))


def prepare_uploaded_reviews(
    dataframe: pd.DataFrame,
    text_column: str,
    title_column: Optional[str] = None,
) -> pd.DataFrame:
    if text_column not in dataframe.columns:
        raise ValueError(f"Review text column '{text_column}' was not found.")

    prepared = dataframe.copy()
    text_values = prepared[text_column].fillna("").astype(str).str.strip()

    if title_column:
        if title_column not in prepared.columns:
            raise ValueError(f"Title column '{title_column}' was not found.")
        title_values = prepared[title_column].fillna("").astype(str).str.strip()
        prepared["app_upload_text"] = (title_values + " " + text_values).str.strip()
    else:
        prepared["app_upload_text"] = text_values

    prepared = prepared[prepared["app_upload_text"] != ""].copy()
    if prepared.empty:
        raise ValueError("Uploaded CSV does not contain any non-empty review text.")

    return prepared
