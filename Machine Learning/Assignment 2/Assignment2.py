import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import numpy as np

# Details for wildfire dataset
datasetFileName = "steel.csv"
independent_cols = ["normalising_temperature","tempering_temperature","percent_silicon","percent_chromium","percent_copper","percent_nickel","percent_sulphur","percent_carbon","percent_manganese"]
dependent_col = "tensile_strength"

# Load in dataset
dataset = pd.read_csv(datasetFileName)

X = np.array(dataset.loc[:,independent_cols])
y = np.array(dataset.loc[:,dependent_col])

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(X):

    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


