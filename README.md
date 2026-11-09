DL_T15 — Machine Temperature Anomaly Detection
Team 15 | Task 3D — Machine Temperature
CM701PC Deep Learning and CM751PC Deep Learning Lab
Department of Emerging Technologies, Mahatma Gandhi Institute of Technology (Autonomous), Hyderabad
Regulation MR-22 | Academic Year 2026
Faculty: Dr. M. Rudra Kumar
Project overview
This project explores unsupervised anomaly detection for an industrial machine-temperature time series using reconstruction-based Autoencoders. Three architectures were compared:
1. Dense Autoencoder baseline (bottleneck size 2)
2. Improved Dense Autoencoder (bottleneck size 1)
3. LSTM Autoencoder (16-dimensional latent dense representation)
Each model reconstructs 12 consecutive sensor readings. Mean squared reconstruction error is used as the anomaly score. Validation data is used to choose thresholds, and the test period is used for evaluation.
Reproducibility and protocol note: the notebook is the submitted experiment record, not yet a final protocol-compliant pipeline. The current training code filters training rows using NAB anomaly-window labels, the sigmoid output is inconsistent with negative scaled values, random seeds are not set, event-level delay is not calculated, and the required PR curve is missing. Read [`docs/NOTEBOOK_AUDIT.md`](docs/NOTEBOOK_AUDIT.md) before citing the results or submitting this experiment.

Repository structure
DL_T15/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── DL_Team15_Machine_Temperature_Anomaly_Detection.ipynb
├── src/
│   ├── data_pipeline.py
│   ├── model_architectures.py
│   ├── evaluation.py
│   └── plotting.py
├── figures/
│   ├── README.md
│   └── *.png
├── results/
│   └── *.csv
├── data/
│   └── README.md
├── docs/
│   ├── NOTEBOOK_AUDIT.md
│   └── EXPERIMENT_NOTES.md
└── report/
    ├── Team15_Machine_Temperature_Anomaly_Detection_Report_DRAFT.docx
    └── Team15_Machine_Temperature_Anomaly_Detection_Report_DRAFT.pdf
Dataset
- Dataset: machine_temperature_system_failure.csv
- Source: Numenta Anomaly Benchmark (NAB)
- CSV: data/realKnownCause/machine_temperature_system_failure.csv
- Event windows: labels/combined_windows.json
- Required data columns: timestamp, value
The notebook downloads the CSV directly from the NAB GitHub raw-content URL. The dataset is not duplicated in this repository. See [`data/README.md`](data/README.md).
Dataset profile in the captured notebook output
Item	Value
Raw observations	22,695
Cleaned observations	22,683
Training observations	13,609
Validation observations	4,537
Test observations	4,537
Exact NAB point labels	4
Observations inside NAB anomaly windows	2,268
Window size	12 readings
Sampling interval	5 minutes


Methodology recorded in the notebook
1. Load and inspect the NAB CSV.
2. Parse timestamps, sort records, drop duplicate timestamps and handle missing numeric values.
3. Load the NAB anomaly-window metadata.
4. Create a chronological 60% / 20% / 20% train-validation-test split.
5. Fit a MinMaxScaler using the training subset.
6. Create sliding windows of 12 observations.
7. Train Dense, Improved Dense and LSTM Autoencoders with Adam and MSE.
8. Compute reconstruction errors.
9. Select best-F1 and cost-based thresholds on validation windows.
10. Evaluate predictions on test windows.
The exact notebook and model definitions are in [`notebooks/`](notebooks/). Reusable helper functions are in [`src/`](src/).
Model architectures
Model	Architecture	Bottleneck / latent representation
Dense AE	12 → 8 → 4 → 2 → 4 → 8 → 12	2
Improved Dense AE	12 → 6 → 3 → 1 → 3 → 6 → 12	1
LSTM AE	LSTM(32) → Dense(16) → Repeat(12) → LSTM(32) → TimeDistributed(Dense(1))	16


Captured experiment results
The values below are transcribed from stored notebook outputs. They are window-level results from the uploaded notebook; they have not been independently rerun in this packaging step.
Best-F1 threshold selected on validation windows
Model	Threshold	Validation precision	Validation recall	Validation F1
Dense AE	0.063619	0.450862	0.904844	0.601841
Improved Dense AE	0.000587	0.335777	0.792388	0.471679
LSTM AE	0.000231	0.300752	0.415225	0.348837


Test metrics at the validation best-F1 thresholds
Model	Precision	Recall	F1	FP windows	FN windows
Dense AE	0.821694	0.956747	0.884093	120	25
Improved Dense AE	0.572295	0.979239	0.722399	423	12
LSTM AE	0.605473	0.918685	0.729897	346	47


Dense AE has the best recorded test F1 in this run. The Improved Dense AE has higher recall but also substantially more false-positive windows. The LSTM AE does not improve the overall balance over the Dense baseline in these saved outputs.
Approximate false-alarm run rates reported by notebook code
Model	Test recall	Alarm runs per day
Dense AE	0.956747	0.190434
Improved Dense AE	0.979239	3.300860
LSTM AE	0.918685	6.284329


These are the recall values from the test metrics table combined with the alarm-run rates produced by the notebook's operational cell. Alarm grouping is a simple contiguous-window count and has not been validated against event matching. It is a preliminary estimate, not a final plant alarm metric.
Cost-based thresholds
The notebook uses the assignment's nominal costs:
- Missed fault: ₹400,000
- False alarm: ₹2,000
Its cost optimisation counts window-level FP and FN values, including overlapping windows:
Model	Validation window cost	Test window cost
Dense AE	₹1,956,000	₹12,394,000
Improved Dense AE	₹6,386,000	₹6,254,000
LSTM AE	₹7,858,000	₹7,866,000


Do not interpret these values as true operational fault/alarm costs until window predictions are merged and matched to fault events. Further details are in [`results/`](results/) and the audit.
Assignment targets — current evidence
Requirement	Target	Current status from saved notebook
Recall on labelled fault events	≥ 0.85	All three test window recalls exceed 0.85; event-level recall not yet verified
False alarms	< 2 per 24 h	Dense estimate <2/day; Improved Dense and LSTM estimates exceed 2/day; event-level validation required
Detection delay	< 5 minutes	Not calculated
Label use	Labels only for threshold tuning and testing	Current notebook uses anomaly-window labels to filter training rows; protocol correction required
Precision-recall curve with both operating points	Required	Current graph is threshold vs F1; PR curve must be added
Cost recommendation	Based on missed-fault and false-alarm events	Current implementation counts windows; event-level costing required


Run the notebook
Google Colab
1. Upload or open notebooks/DL_Team15_Machine_Temperature_Anomaly_Detection.ipynb in Google Colab.
2. Connect to a runtime.
3. Run cells from top to bottom.
4. The notebook fetches the NAB CSV and combined anomaly-window labels from the NAB GitHub repository.
5. Save a copy of the executed notebook with the outputs.
Local environment
Python 3.10 or later is recommended.
python -m venv .venv
Activate the environment, then install dependencies:
pip install -r requirements.txt
Launch Jupyter:
jupyter notebook
Open the notebook from notebooks/.
Required corrections before final submission
See [`docs/NOTEBOOK_AUDIT.md`](docs/NOTEBOOK_AUDIT.md). In brief:
1. Redesign training-period selection so NAB labels are not used to filter training observations.
2. Ensure training windows remain temporally contiguous.
3. Make the scaler's output range compatible with the decoder activation.
4. Set and record random seeds.
5. Recompute a final test-only metrics table.
6. Add the required precision-recall curve with best-F1 and cost-optimal points.
7. Calculate event-level recall, false alarms per day and detection delay.
8. Compute operational cost per missed fault event and false-alarm event, not per overlapping window.
Team 15
Roll number	Role	Member name
24265A6604	Data Lead	M Bharath
23261A6665	Model Lead	Y Venkata Ramana
23261A6616	Evaluation & Report Lead	Harinayan Raj Pattun


All team members are expected to contribute to code, experiments and report preparation.
Project links
- GitHub repository: https://github.com/Bharath-723/DL_T15
- Colab notebook: [Add Colab link]
- Final report: [`report/`](report/)
AI assistance declaration
AI coding assistance was used during code development and documentation drafting. The team must review, understand, test and disclose the exact assistance used in the final submission, consistent with the department instructions. Update the declaration to reflect the tools actually used by the team.
References
1. Numenta, Numenta Anomaly Benchmark (NAB), GitHub repository: https://github.com/numenta/NAB
2. I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. Cambridge, MA: MIT Press, 2016, ch. 14.
