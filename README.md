# Kenya Rainfall Forecasting and Anomaly Classification

This project uses recurrent neural networks, specifically Long Short-Term Memory (LSTM) models, to analyze monthly rainfall patterns in Kenya from 1991 to 2016. It was built as a neural networks assignment and is structured as a small, reproducible machine learning workflow rather than a single model-training script.

The project includes two related tasks:

- Forecasting next-month rainfall amount from the previous 12 months of rainfall history.
- Classifying next-month rainfall as below-normal, normal, or above-normal using anomaly thresholds derived from the training period.

## Project Highlights

- Cleaned and transformed raw year/month rainfall records into a chronological monthly time series.
- Converted the time series into 12-month sliding windows for LSTM training.
- Engineered rainfall and seasonality features including rolling rainfall averages, anomaly z-scores, rainfall ratios, first-order rainfall change, and cyclic month encodings.
- Used chronological train, validation, and test splits so future observations do not leak into model training.
- Added baseline classifiers such as majority class, previous-month persistence, and target-month majority to judge whether the LSTM adds value.
- Evaluated classification with accuracy, balanced accuracy, macro F1, confusion matrices, and per-class precision/recall rather than relying on accuracy alone.

## Repository Structure

```text
rainfall_RNN_lstm_project/
|-- data/
|   `-- kenya-climate-data-1991-2016-rainfallmm.csv
|-- notebooks/
|   |-- Rainfall_Prediction_RNN.ipynb
|   `-- kenya_rainfall_lstm.ipynb
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Notebooks

### `notebooks/Rainfall_Prediction_RNN.ipynb`

Builds a rainfall regression model. It scales monthly rainfall values, creates 12-month input windows, trains an LSTM model, and predicts the next month's rainfall amount.

Saved result:

- Test RMSE: approximately `32.04 mm`

### `notebooks/kenya_rainfall_lstm.ipynb`

Builds a rainfall anomaly classifier. It labels rainfall into three classes:

- `0`: below-normal anomaly
- `1`: normal
- `2`: above-normal anomaly

The notebook currently uses `TARGET_MODE = "monthly_anomaly"`, which compares each month against historical values for the same calendar month. This is a stricter climatological anomaly task than using one global rainfall threshold for all months.

The classifier uses:

- 12 months of rainfall history
- engineered rainfall features
- target-month cyclic encodings
- an LSTM layer followed by dense classification layers

## Current Results

The month-relative anomaly task is intentionally difficult because the dataset contains only 26 years of monthly observations. The notebook therefore includes baseline models to avoid overstating LSTM performance.

Current anomaly-classification test period: `2012-01-01` to `2016-12-01`

| Model | Accuracy | Balanced Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Majority class | 0.167 | 0.333 | 0.095 |
| Previous-month persistence | 0.400 | 0.389 | 0.389 |
| Target-month majority | 0.167 | 0.333 | 0.095 |
| LSTM classifier | 0.367 | 0.400 | 0.284 |

The persistence baseline remains competitive, which is useful diagnostically: it shows that this project does not just train a neural network, but also checks whether the model improves on simple time-series heuristics.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then open the notebooks with Jupyter:

```bash
jupyter notebook
```

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Notes

The repository intentionally excludes local virtual environments, notebook checkpoints, and generated HTML exports. The source dataset is small and included so the notebooks can be run without needing an external download.
