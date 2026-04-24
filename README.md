# Heart Attack / MI Complications Prediction

This Project uses the UCi Myocardial Infarction Complications dataset to train a first machine learning model.
https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications

## Files

- `heart_attack_prediction.py`
    Main script that loads data, cleans it, trains models, and evaluates them

- `src/load_data.py`
    Loads the dataset from UCI using 'ucimlrepo'

- `src/preprocess.py`
    Selects the target, drops bad columns, and splits data

- `src/train_model.py`
    Builds preprocessing and achine learning pipelines

- `src/evaluate.py`
    Prints accuracy, precision, recall, F1, ROC-AUC, and confusion matrix

## How to Run

```bash
pip install -r requirements.txt
python heart_attack_prediction.py
