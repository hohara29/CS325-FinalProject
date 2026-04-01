from ucimlrepo import fetch_ucirepo
import pandas as pd

def load_uci_mi_data():
    #Loads the UCI Myocardial Infarction Complications dataset.

    #Returns:
    #    X (pd.DataFrame): input features
    #    y_all (pd.DataFrame): all target/output columns
    
    mi = fetch_ucirepo(id=579)

    X = mi.data.features.copy()
    y_all = mi.data.targets.copy()

    return X, y_all