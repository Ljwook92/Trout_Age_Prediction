# Trout_Age_Prediction
Age Prediction by using Trout scale

## Local data split

Use three separate CSVs when running the expert-feedback workflow:

- `review.csv`: images shown to experts for `Correct` / `Incorrect` feedback.
- `validation.csv`: fixed labeled set used after every 20 new feedbacks to decide whether a fine-tuned candidate becomes the best model.
- `test.csv`: final holdout set for reporting only. Do not show it in the app and do not use it for candidate selection.

Configure paths with environment variables:

```bash
export TROUT_REVIEW_CSV_PATH=/path/to/review.csv
export TROUT_VALIDATION_CSV_PATH=/path/to/validation.csv
export TROUT_TEST_CSV_PATH=/path/to/test.csv
streamlit run app.py
```

If `TROUT_VALIDATION_CSV_PATH` is not set, the app falls back to `TROUT_REVIEW_CSV_PATH` and uses `source == "labeled"` plus `streamlit == 0` rows when available. For clean evaluation, prefer a separate validation CSV.
