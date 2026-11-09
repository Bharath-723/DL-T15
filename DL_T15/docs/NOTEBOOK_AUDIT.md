# Notebook review and reproducibility audit

This audit is based on the uploaded `DL_Team15_Machine_Temperature_Anomaly_Detection.ipynb`, including its stored cell outputs. It distinguishes observed notebook output from methodological issues inferred from the code. No cells were executed as part of preparing this repository package.

## 1. What the notebook currently contains

- 51 code cells, with captured outputs and 12 embedded plots.
- Data source: NAB `realKnownCause/machine_temperature_system_failure.csv`.
- Raw data reported: 22,695 rows; cleaned data reported: 22,683 rows.
- Sampling cadence shown in the first five rows: 5 minutes.
- NAB point labels: four timestamps.
- NAB anomaly intervals: four intervals covering 2,268 observations in the cleaned series.
- Chronological split: 13,609 train, 4,537 validation, 4,537 test observations.
- Window length: 12 readings (one hour at 5-minute sampling).
- Models: Dense AE bottleneck 2; Improved Dense AE bottleneck 1; LSTM AE latent Dense(16).
- Training settings visible in cells: Adam, MSE, batch size 32, maximum 50 epochs, early stopping with patience 5 and best-weight restoration.

## 2. Captured model results

The following are the notebook's stored outputs, not an independent rerun.

### Best-F1 thresholds selected on validation windows

| Model | Threshold | Validation precision | Validation recall | Validation F1 |
|---|---:|---:|---:|---:|
| Dense AE | 0.063619 | 0.450862 | 0.904844 | 0.601841 |
| Improved Dense AE | 0.000587 | 0.335777 | 0.792388 | 0.471679 |
| LSTM AE | 0.000231 | 0.300752 | 0.415225 | 0.348837 |

### Test metrics at the validation best-F1 thresholds

| Model | Precision | Recall | F1 | FP windows | FN windows |
|---|---:|---:|---:|---:|---:|
| Dense AE | 0.821694 | 0.956747 | 0.884093 | 120 | 25 |
| Improved Dense AE | 0.572295 | 0.979239 | 0.722399 | 423 | 12 |
| LSTM AE | 0.605473 | 0.918685 | 0.729897 | 346 | 47 |

The separate false-alarm calculation in notebook cell 41 reports approximately 0.1904, 3.3009 and 6.2843 alarm runs/day for Dense, Improved Dense and LSTM respectively. The recall values displayed next to those rates in cell 42 are validation recall, whereas the false-alarm rates are calculated from test predictions. Therefore cell 42 mixes metrics from different splits. The test recalls in the table above are the matching test values.

### Cost-threshold results

The cost function uses ₹400,000 per false negative and ₹2,000 per false positive, but counts individual overlapping windows:

| Model | Validation cost (window-level) | Test cost (window-level) |
|---|---:|---:|
| Dense AE | ₹1,956,000 | ₹12,394,000 |
| Improved Dense AE | ₹6,386,000 | ₹6,254,000 |
| LSTM AE | ₹7,858,000 | ₹7,866,000 |

These amounts should not be reported as plant-operational costs. Overlapping windows from a single fault or alarm are correlated observations; the assignment's costs refer to a missed fault and a false alarm. The final operational cost should be recomputed after grouping detections into alarm events and matching those events to NAB fault windows.

## 3. Issues to fix before claiming assignment compliance

### A. Training uses ground-truth anomaly-window labels

Notebook cells 8–10 filter `train_df` with `~train_df["in_anomaly_window"]` before fitting the scaler and building training windows. This uses NAB labels to decide which training records to exclude. The brief states that labels are reserved for threshold tuning and testing. As written, this experiment does not strictly meet that protocol. Do not describe it as fully label-free training until the team redesigns and reruns the training-period selection without consulting the labelled fault intervals.

### B. Removing labelled rows before windowing breaks temporal continuity

After anomaly rows are removed, `create_sequences` is applied to the remaining concatenated values. This can create a 12-reading sequence that joins observations from before and after a removed fault interval, even though those readings were not adjacent in real time. A revised training-window builder should preserve the original chronological stream and avoid creating windows across gaps. This should be addressed together with the label-use issue above.

### C. Scaling range and sigmoid output are inconsistent

The notebook fits `MinMaxScaler` on normal training values and transforms the full dataset. Its captured output reports a scaled minimum of approximately -0.8964. However, all three decoder outputs use sigmoid activation, which is constrained to [0, 1]. The models cannot reconstruct negative scaled values. Use a compatible scaling/output design (for example, a linear reconstruction output when the target range can extend outside [0, 1]) and rerun the models before using the reported reconstruction errors as final evidence.

### D. Validation loss includes anomalous windows

The models use the entire validation set in `validation_data` for early stopping, including windows that overlap NAB anomaly intervals. This does not directly train on labels, but validation loss is influenced by anomalous samples. A revised experiment should state the model-selection protocol clearly and consider a clean, label-independent validation strategy.

### E. Reproducibility seed is not set

The notebook does not set global NumPy and TensorFlow seeds. Add fixed seeds near the start, e.g. `np.random.seed(42)` and `tf.keras.utils.set_random_seed(42)`, then rerun and save the resulting notebook.

### F. Recall and false-alarm table mixes data splits

Notebook cell 42 uses `best_*['recall']` from validation and false-alarm/day values calculated from test predictions. Rebuild a final test-only operational table from `test_results` and the same test prediction arrays.

### G. False alarms/day and detection delay are not fully evaluated

The notebook counts starts of contiguous positive windows after masking windows labelled as anomalous. It does not perform full matching of alarm episodes to each of the four NAB fault intervals. Detection delay after each event start is not calculated. The assignment specifically requires fewer than two false alarms per 24 hours and a delay under five minutes (or the stated exception). These requirements cannot yet be fully verified from the notebook outputs.

### H. Required precision-recall plot is missing

Notebook cell 35 plots threshold against F1. The assignment asks for a precision-recall curve and both the best-F1 and cost-optimal operating points marked on it. Add that plot using validation predictions and export it.

## 4. Current status against the brief

| Requirement | Current evidence | Status |
|---|---|---|
| Recall ≥ 0.85 | Test-window recall is ≥ 0.9187 for all three models at best-F1 thresholds | Meets on current window-level test metrics; event-level recall still needed |
| False alarms < 2 per 24 h | Existing alarm-run estimate: Dense 0.1904/day; Improved 3.3009/day; LSTM 6.2843/day | Dense appears below target; event-level validation needed |
| Detection delay < 5 min | No event-level delay calculation in notebook | Not evaluated |
| Labels only for threshold tuning and testing | NAB labels used to filter training rows | Not met by current pipeline |
| PR curve with best-F1 and cost thresholds | Threshold-F1 plot only | Missing |
| Event-based cost recommendation | Current cost counts windows | Not valid as operational cost yet |
| Fixed random seed | Not set | Missing |

## 5. Suggested correction order

1. Decide and document a label-independent training-period selection method.
2. Rebuild training windows without joining across removed temporal intervals.
3. Use a scaler and decoder output activation that are mathematically compatible.
4. Set fixed seeds and rerun all three models.
5. Recompute validation thresholds and the required PR curve.
6. Evaluate test data only once using the selected thresholds.
7. Merge positive windows into alarm episodes, match alarms to NAB fault intervals, and calculate event recall, false alarms/day and detection delay.
8. Recompute ₹ cost using fault events and false-alarm events, then update the report and README.

## 6. Scope of this repository

The notebook in `notebooks/` is retained as the submitted experiment record. The files in `src/` provide reusable helper functions matching the current notebook's basic preprocessing, architectures and window-level evaluation. They are not a corrected or independently validated end-to-end pipeline. The CSV result tables are transcribed from saved notebook outputs; they are not fresh model runs.
