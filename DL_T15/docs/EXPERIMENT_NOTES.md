# Experiment notes

## Dataset and split recorded in the notebook
- Raw CSV: 22,695 rows.
- After cleaning: 22,683 rows.
- Train / validation / test: 13,609 / 4,537 / 4,537 observations.
- NAB point labels: 4 exact timestamps.
- NAB anomaly windows: 4 intervals, 2,268 rows marked inside an interval.
- Window size: 12 readings, approximately 60 minutes at the 5-minute cadence.

## Model configurations recorded
- Dense AE: 12 → 8 → 4 → 2 → 4 → 8 → 12.
- Improved Dense AE: 12 → 6 → 3 → 1 → 3 → 6 → 12.
- LSTM AE: LSTM(32) → Dense(16) → RepeatVector(12) → LSTM(32) → TimeDistributed(Dense(1)).
- Optimizer: Adam; loss: MSE; batch size: 32; max epochs: 50; early stopping patience: 5.

## Key observation from saved test output
At the validation best-F1 thresholds, Dense AE has the highest recorded test F1 (0.884093), while Improved Dense AE and LSTM AE have higher recall but materially more false-positive windows. This supports the conclusion that the narrower bottleneck / sequential architecture did not improve the overall window-level precision-recall balance in this run.

See `NOTEBOOK_AUDIT.md` for methodological issues that must be resolved before final submission.
