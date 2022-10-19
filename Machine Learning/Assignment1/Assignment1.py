import pandas as pd
import DecisionTreeClassifier
import AdaBoostClassifier


# Details for wildfire dataset
training_file = "wildfires_training.csv"
test_file = "wildfires_test.csv"
independent_cols = ["year","temp","humidity","rainfall","drought_code","buildup_index","day","month","wind_speed"]
dependent_col = "fire"

# Load in training dataset in from the training file using the pandas library
df_training = pd.read_csv(training_file)
print(df_training.head())
print(df_training.shape)

# Set up a matrix X containing the independent variables from the training data
X_training = df_training.loc[:,independent_cols]
print(X_training.head())
print(X_training.shape)

# Set up a vector y containing the dependent variable / target attribute for the training data
y_training = df_training.loc[:,dependent_col]
print(y_training.head())
print(y_training.shape)

# Next we load our test dataset in from the file wildfires_test.csv
df_test = pd.read_csv(test_file)
print(df_test.head())
print(df_test.shape)

# set up a matrix X containing the independent variables from the test data
X_test = df_test.loc[:,independent_cols]
print(X_test.head())
print(X_test.shape)

# Set up a vector y containing the dependent variable / target attribute for the training data
y_test = df_test.loc[:,dependent_col]
print(y_test.head())
print(y_test.shape)


DecisionTreeClassifier.DecisionTreeClassifierMethod(X_training, y_training, X_test, y_test)
AdaBoostClassifier.AdaBoostClassifierMethod(X_training, y_training, X_test, y_test)








