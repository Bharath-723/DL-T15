# Dataset

The raw NAB machine-temperature CSV is intentionally not bundled in this repository. The notebook downloads it from the official Numenta Anomaly Benchmark (NAB) GitHub raw URL:

`https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv`

The NAB anomaly-window metadata is read from:

`https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json`

The file used in the submitted notebook has 22,695 raw rows. The notebook cleaning cell reports 22,683 rows after sorting, duplicate-timestamp removal and missing-value handling. The notebook's chronological split reports 13,609 training rows, 4,537 validation rows and 4,537 test rows.

Do not commit a second copy of the source CSV unless specifically required. The dataset is retrieved at runtime. The official source and label definitions should be cited in the report/README.
