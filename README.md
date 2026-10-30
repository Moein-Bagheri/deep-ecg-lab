# Deep ECG lab

**93% accuracy can hide a lot.** A two-layer GRU classifies heartbeats from ECG signals. The headline number is high, but the model recognizes only 6% of supraventricular beats.

<img src="outputs/mitbih_confusion_matrix.png" width="480" alt="MIT-BIH confusion matrix: normal, ventricular and unknown beats are mostly recognized; supraventricular beats are mostly predicted as normal">

*TensorFlow · Keras · scikit-learn*

## What I did

- Built a two-layer GRU (64 and 32 units, dropout 0.2) for heartbeat sequences.
- Trained it on MIT-BIH for five beat classes (normal, supraventricular, ventricular, fusion, unknown) and on PTBDB for normal against abnormal.
- Saved both trained models, their confusion matrices and training curves.

## What I found

| Task | Test beats | Accuracy | Recall by class |
| --- | ---: | ---: | --- |
| MIT-BIH, five classes | 21,892 | 93.3% | normal 97% · unknown 93% · ventricular 89% · fusion 35% · supraventricular 6% |
| PTBDB, normal or abnormal | 2,911 | 85.5% | abnormal 96% · normal 59% |

- **Imbalance wins.** Normal beats are 83% of MIT-BIH, and 481 of 556 supraventricular beats are called normal.
- **On PTBDB the model leans toward "abnormal".** It catches 96% of abnormal beats, but flags 41% of normal ones too.
- What I'd try next: class weights or resampling, and reporting macro F1 beside accuracy.

## Try it

The trained models and figures are in `outputs/`, so you can look at the results without the data. To retrain, put the MIT-BIH and PTBDB CSV files in `data/`:

```text
data/mitbih_train.csv   data/mitbih_test.csv
data/ptbdb_normal.csv   data/ptbdb_abnormal.csv
```

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main_mitbih.py
.\.venv\Scripts\python.exe main_ptbdb.py
```

## Layout

| Path | What's there |
| --- | --- |
| [`main_mitbih.py`](main_mitbih.py) | Five-class heartbeat model |
| [`main_ptbdb.py`](main_ptbdb.py) | Normal or abnormal model |
| [`outputs/`](outputs/) | Saved models, confusion matrices and training curves |

Deep learning course project, University of Isfahan, 2026.
