import unittest

import pandas as pd

from uploaded_reviews import prepare_uploaded_reviews, review_column_candidates


class UploadedReviewsTest(unittest.TestCase):
    def test_adds_xquik_columns_after_existing_candidates(self):
        candidates = review_column_candidates(["review_text", "comment"])

        self.assertEqual(candidates[:2], ["review_text", "comment"])
        self.assertIn("full_text", candidates)
        self.assertIn("tweet_text", candidates)

    def test_combines_title_and_review_text(self):
        dataframe = pd.DataFrame(
            {
                "headline": ["Delivery"],
                "full_text": ["Arrived earlier than expected"],
            }
        )

        prepared = prepare_uploaded_reviews(
            dataframe,
            text_column="full_text",
            title_column="headline",
        )

        self.assertEqual(prepared["app_upload_text"].to_list(), ["Delivery Arrived earlier than expected"])

    def test_drops_blank_uploaded_rows(self):
        dataframe = pd.DataFrame({"tweet_text": ["Great", " ", None, "Slow"]})

        prepared = prepare_uploaded_reviews(dataframe, text_column="tweet_text")

        self.assertEqual(prepared["app_upload_text"].to_list(), ["Great", "Slow"])

    def test_rejects_empty_uploads(self):
        dataframe = pd.DataFrame({"review_text": [" ", None]})

        with self.assertRaises(ValueError):
            prepare_uploaded_reviews(dataframe, text_column="review_text")


if __name__ == "__main__":
    unittest.main()
