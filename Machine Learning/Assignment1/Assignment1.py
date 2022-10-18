import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt


# details for wildfire dataset
training_file = "wildfires_training.csv"
test_file = "wildfires_test.csv"
independent_cols = ["year","temp","humidity","rainfall","drought_code","buildup_index","day","month","wind_speed"]
dependent_col = "fire"

# Here we load our training dataset in from the training file using the pandas library
df_training = pd.read_csv(training_file)
print(df_training.head())
print(df_training.shape)

# set up a matrix X containing the independent variables from the training data
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


# create a model for DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_training, y_training)

# compute the predictions for the training and test sets
predictions_training = model.predict(X_training)
predictions_test = model.predict(X_test)

# compute the accuracy on the training and test set predictions
accuracy_training = metrics.accuracy_score(y_training, predictions_training)
accuracy_test = metrics.accuracy_score(y_test, predictions_test)
print("Accuracy on training data:",accuracy_training)
print("Accuracy on test data:",accuracy_test)


# Now let's evaluate the effect of using different k values
# start at k=1 and test all odd k values up to 21
k_values = list(range(1,31,2))
j_values = ["gini", "entropy", "log_loss"]

print(k_values)

accuracy_training_k = []
accuracy_test_k = []

for k in k_values:
    for j in j_values:
        model_k = DecisionTreeClassifier(max_depth = k, criterion = j)
        model_k.fit(X_training, y_training)

        # compute the predictions for the training and test sets
        predictions_training_k = model_k.predict(X_training)
        predictions_test_k = model_k.predict(X_test)

        # compute the accuracy on the training and test set predictions
        accuracy_training_k.append(metrics.accuracy_score(y_training, predictions_training_k))
        accuracy_test_k.append(metrics.accuracy_score(y_test, predictions_test_k))

print(accuracy_training_k)
print(accuracy_test_k)

# let's plot the accuracy on the training and test set
plt.scatter(k_values,accuracy_training_k,marker="x")
plt.scatter(k_values,accuracy_test_k,marker="+")
plt.xlim([0, max(k_values)+2])
plt.ylim([0.0, 1.1])
plt.xlabel("Value of k")
plt.ylabel("Accuracy")
legend_labels = ["Training (Euclidian dist.)","Test (Euclidian dist.)"]
plt.legend(labels=legend_labels, loc=4, borderpad=1)
plt.title("Effect of k on training and test set accuracy", fontsize=10)
plt.show()