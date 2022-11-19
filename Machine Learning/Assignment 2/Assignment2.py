import numpy as np
import pandas as pd
from KNeighbours import KNeighboursRegressorFunction
from DecisionTreeRegression import DecisionTreeRegressionFunction


# Details for steel dataset
datasetFileName = "steel.csv"
independent_cols = ["normalising_temperature","tempering_temperature","percent_silicon","percent_chromium","percent_copper","percent_nickel","percent_sulphur","percent_carbon","percent_manganese"]
dependent_col = "tensile_strength"

# Load in dataset
dataset = pd.read_csv(datasetFileName)

# Create 2 arrays which contain the independent and dependent variables
X = np.array(dataset.loc[:,independent_cols])
y = np.array(dataset.loc[:,dependent_col])

KNeighboursRegressorFunction(X,y)
DecisionTreeRegressionFunction(X,y)



