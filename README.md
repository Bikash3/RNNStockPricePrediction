# Stock Price Prediction Using RNNs
> Time-series forecasting of stock closing prices for selected technology companies using recurrent neural network architectures.

## Table of Contents
* [General Info](#general-information)
* [Stepwise Process](#stepwise-process)
  * [Step 1: Objective](#step-1---objective)
  * [Step 2: Data Understanding](#step-2---data-understanding)
  * [Step 3: Data Loading and Preparation](#step-3---data-loading-and-preparation)
  * [Step 4: RNN Models and Tuning](#step-4---rnn-models-and-tuning)
  * [Step 5: Predicting Multiple Targets (optional)](#step-5---predicting-multiple-targets-optional)
  * [Step 6: Conclusions and Insights](#step-6---conclusions-and-insights)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Info
> This repository implements a pipeline to forecast equity closing prices using historical daily data from four technology companies. The workflow covers data aggregation, exploratory analysis, temporal window creation, scaling, model building (Simple RNN, LSTM, GRU), hyperparameter search and evaluation.

Target stocks used:
- Amazon (AMZN)
- Google / Alphabet (GOOGL)
- IBM (IBM)
- Microsoft (MSFT)

## Stepwise Process

### Step 1 - Objective
- Build models to predict future closing prices using historical sequences.
- Use combined data from multiple related stocks to improve generalization.
- Provide reproducible steps for data preparation, model training, tuning and evaluation.

### Step 2 - Data Understanding
- Input files: CSVs for each stock with identical columns.
- Columns:
  - Date, Open, High, Low, Close, Volume, Name
- Date range used in examples: 2006-01-01 to 2018-01-01 (3019 records per file in the provided data).
- Key checks:
  - Missing values
  - Per-stock row counts
  - Date parsing and chronological order

### Step 3 - Data Loading and Preparation
1. Aggregate CSV files into a single DataFrame (preserve Name column for provenance).
2. Handle missing values (drop or impute as appropriate).
3. Exploratory plots:
   - Volume distributions per stock
   - Volume variation over time
   - Resampled trends (weekly, monthly, quarterly) for Close prices
   - Correlation heatmap for numeric features
4. Windowing:
   - Create sliding windows of past timesteps to predict next-period Close.
   - Parameters: window_size, stride (window step), feature columns, target names.
   - Example: window_size = 21, stride = 1 used in notebook.
5. Scaling:
   - Scale windowed X and y (example uses MinMaxScaler fitted on windowed data).
   - Avoid leakage by fitting scalers appropriately for the windowing approach.
6. Train / test split:
   - Use time-ordered split (shuffle=False) with test_size (example 0.3).

### Step 4 - RNN Models and Tuning
1. Model types implemented and experimented:
   - SimpleRNN
   - LSTM
   - GRU
2. Model building:
   - Define reusable builders that accept configuration: time_steps, features, units, layers, dropout, optimizer, loss, output dimension.
   - Example SimpleRNN pipeline: SimpleRNN -> Dense(output)
3. Training strategy:
   - Early stopping and model checkpoints where applicable.
   - Use validation split during training for tuning.
4. Hyperparameter search:
   - Use Keras Tuner (Hyperband or RandomSearch) to tune units, learning rate, dropout, and RNN type.
   - Objective examples: val_loss or val_mae.
5. Evaluation:
   - Plot actual vs predicted series.
   - Report test metrics (MSE, MAE).
   - Example notebook flows include: tune simple RNN, retrain best model, evaluate on X_test/y_test.

### Step 5 - Predicting Multiple Targets (optional)
- Support multi-target forecasting: predict Close for multiple stocks simultaneously.
- Adjust model output dimension and loss accordingly (e.g., MSE for regression, no activation on final Dense).
- Windowing and scaling must accommodate multi-output y shapes.

### Step 6 - Conclusions and Insights
- Summarize findings: performance differences across model families, impact of window size, and usefulness of multi-stock input.
- Typical observations:
  - Proper window selection (weekly/monthly/quarter patterns) affects predictive power.
  - Advanced cells (LSTM/GRU) often handle longer dependencies better than SimpleRNN.
  - Hyperparameter tuning yields measurable improvements; monitor validation to avoid overfitting.
- Recommendations:
  - Build a hold-out evaluation set for final model validation.
  - Use walk-forward validation for time-series robustness.
  - Add feature engineering (technical indicators) and alternative loss functions if objective requires.

## Technologies Used
- Python
- NumPy, pandas
- Matplotlib, Seaborn
- scikit-learn (MinMaxScaler, train_test_split)
- TensorFlow / Keras (SimpleRNN, LSTM, GRU)
- Keras Tuner
- Jupyter Notebook

## Acknowledgements
- Notebook prepared as part of course exercises and combined work by contributors.
- Reference material:
  - https://www.tensorflow.org/
  - https://scikit-learn.org/
  - https://keras.io/
  - https://keras-team.github.io/keras-tuner/

## Contact
### Created by
  * Bikash Sarkar