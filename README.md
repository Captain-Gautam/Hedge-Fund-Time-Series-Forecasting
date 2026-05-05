# Hedge Fund - Time Series Forecasting (Kaggle Competition)

## Overview
This repository contains the code and methodology for the Kaggle Competition: **Hedge Fund - Time Series Forecasting**. The task is to predict future financial market values using historical observations while avoiding any look-ahead bias.

Given the extreme noise and heavy-tailed distribution characteristic of financial data, predicting a directional signal is highly challenging. The baseline of predicting zero is notoriously hard to beat. Our final approach utilizes a custom **Temporal Fusion Transformer (TFT)** augmented with **Quantile Regression** (Predicting the 10th, 50th, and 90th percentiles) to output robust probabilistic forecasts that successfully beat the zero-prediction baseline across all targets.

## Problem Statement
Given historical observations \(\{y_1, y_2, \dots, y_t\}\), predict the future value \(y_{t+h}\) at given horizon \(h\). 
- Constraint: No look-ahead bias (strict sequential prediction using data only up to \(t\)).
- The model must generalize to unseen market conditions and regimes.

## Dataset & Preprocessing

- **Dataset Size**: ~5.3 million training rows, ~1.4 million testing rows.
- **Features**: Initially over 80 numeric features, 6 categorical columns (like `code`, `sub_code`, `sub_category`, `horizon`). 
- **Target**: `y_target` with associated `weight` to represent sample importance. Targets range near \(\pm800\) with severe heavy tails.

### Feature Engineering
- **Feature Selection**: Started with 80+ numerics and selected the **Top 30** utilizing Weighted Absolute Pearson Correlation.
- **Sequence Construction**: A sliding window approach with a `Sequence Length = 4`. A memory-efficient reservoir sampling method was applied.
- **Transformations**: Missing value median imputation, StandardScaler for numerics (fitted strictly on training split), and integer encoding for categoricals.

## Methodology & Architecture

The core architecture is based on the **Temporal Fusion Transformer (TFT)** consisting of ~308K parameters.

### Key Components
1. **Variable Selection Network (VSN)**: Generates softmax attention weights to select the most useful features. Guided by static context.
2. **Gated Residual Network (GRN) & Gated Linear Units (GLU)**: Skip connections and gating for stable deep training.
3. **LSTM Encoder**: Captures temporal patterns across the 4-step lookback window.
4. **Multi-Head Attention**: 8 attention heads with a causal mask to prevent data leakage. Captures momentum, volatility, and mean-reversion.

### Quantile (Pinball) Loss
We shifted from simplistic Point Prediction (MSE) to Probabilistic Forecasting utilizing the Pinball Loss for quantiles \(q \in \{0.1, 0.5, 0.9\}\):
- **P10**: Downside risk estimate
- **P50**: Point prediction (median)
- **P90**: Upside potential estimate
*Approximately 80% of targets are expected to fall within the `[P10, P90]` bounds.*

## Iterative Development
The model evolved across 6 major versions:
- **V1 (Baseline)**: Suffered from catastrophic overfitting due to noisy features and aggressive target scaling.
- **V2 (Feature Filter)**: Shrunk to 30 features. Model struggled with scaling flaws that paralyzed gradients.
- **V3 (Convergence)**: Removed artificial target normalization/penalties. Finally achieved raw, unconstrained textbook convergence.
- **V4 (8-Head Attention Expansion)**: Increased attention heads from 2 to 8, allowing the model to track distinct market patterns simultaneously.
- **V6 (Quantile Regression Upgrade)**: Changed output shape to `[B, 3]` and introduced Weighted Pinball Loss, successfully predicting intervals with reliable uncertainty bounds.

## Training Configuration
- **Seq Length**: 4
- **Batch Size**: 1024
- **Hidden Dim**: 48 & 8 Attention Heads
- **Optimizer**: AdamW (LR: \(10^{-3}\), Weight Decay: \(10^{-3}\))
- **Scheduler**: 3 epochs warmup, cosine annealing.
- **Epochs**: 40

## Results & Evaluation (5-Fold CV)
Due to the absence of the test ground truth after the competition, we relied on a robust **5-Fold Temporal Cross-Validation** strategy.
- **Weighted Coverage**: 80.0% (\(\pm5.4\%\)) vs Expected 80%.
- **Competition Score**: 0.04242 (\(\pm0.02107\)). (*Score > 0 means beating the 0-baseline*).
- **Quantile Check**: All three quantiles (`P10`, `P50`, `P90`) consistently beat the zero-prediction baseline across folds.

This demonstrates that the model successfully learned real predictive signal and successfully prioritizes high-importance targets.

## Repository Contents

- `hedge-fund-time-series-forecasting.ipynb`: Initial data exploration, analysis (EDA), and basic pipeline creation.
- `tft_training.ipynb`: Basic Temporal Fusion Transformer training code (covering early versions V1-V4 for point predictions).
- `quantile_training.ipynb`: The main notebook containing the V6 architecture implementation, implementing the multi-head attention and Pinball loss for quantiles `[P10, P50, P90]`.
- `kfold_cv_quantile.ipynb`: Implements the 5-fold Temporal Cross-Validation strategy to robustly estimate competition performance.
- `ppt/presentation.pdf`: Presentation slides summarizing the analysis, findings, methodology, and visualizations of the results.

## Future Work
- **Target Normalization**: Safely scale targets to \(\pm1\) strictly for training and inversely un-scale at inference to combat magnitude mismatch.
- **Ensemble Strategies**: Average top checkpoint predictions for high stability.
- **Hyperparameter Search**: Investigate deeper architectures and longer lookback sequence lengths.
