# Python Data Science Study Notes

## Data Analysis Workflow

A strong data science workflow moves from question to evidence:

1. Define the business or research question.
2. Load data with reproducible code.
3. Inspect schema, missingness, and unusual values.
4. Clean and transform features.
5. Explore patterns with grouped summaries and visualizations.
6. Train baseline models before complex models.
7. Evaluate with metrics aligned to the goal.
8. Communicate findings with assumptions and limitations.

## NumPy

NumPy arrays support vectorized numerical computation. Vectorization is faster
than Python loops because operations run in optimized compiled code.

Key ideas:

- Shape controls how arrays align.
- Broadcasting expands compatible dimensions without copying data.
- Boolean masks filter arrays by conditions.
- Random seeds make simulations reproducible.

## pandas

pandas DataFrames are table-like structures with labeled rows and columns.

Common operations:

- `read_csv` loads tabular data.
- `df.info()` checks types and missing values.
- `groupby` aggregates by category.
- `merge` joins datasets on keys.
- `assign` creates new columns in a readable pipeline.

## Modeling

Regression predicts continuous values. Classification predicts categories.

Good practice:

- Split data into train and test sets before learning patterns.
- Fit preprocessing steps only on the training data.
- Compare against a simple baseline.
- Watch for leakage, where future or target information appears in features.
- Use cross-validation when data is limited.

## Interpretation

A model should be judged not only by accuracy, but also by whether its errors are
acceptable for the real decision. Explain which features matter, where the model
fails, and what data would improve confidence.

