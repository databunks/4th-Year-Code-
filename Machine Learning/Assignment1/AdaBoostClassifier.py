from statistics import mean
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

def AdaBoostClassifierMethod(X_training, y_training, X_test, y_test):

    # create a model for AdaBoostClassifier
    model = AdaBoostClassifier()
    model.fit(X_training, y_training)

    # compute the predictions for the training and test sets
    predictions_training = model.predict(X_training)
    predictions_test = model.predict(X_test)

    # compute the accuracy on the training and test set predictions
    accuracy_training = metrics.accuracy_score(y_training, predictions_training)
    accuracy_test = metrics.accuracy_score(y_test, predictions_test)
    print("Accuracy on training data:",accuracy_training)
    print("Accuracy on test data:",accuracy_test)

    # Now let's evaluate the effect of using different k values for the hyperparameter n_estimators
    # start at k=1 and test all odd k values up to 31
    k_values = list(range(1,31,2))

    # j values for algorithm hyperparameter
    j_values = ["SAMME", "SAMME.R"]

    print("n_estimators: " + str(k_values))
    print("algorithm values: " + str(j_values))

    accuracy_training_k = []
    accuracy_test_k = []

    # Iterating through each j value and k value and producing graphs for the result
    for j in j_values:
            for k in k_values:
                    model_k = AdaBoostClassifier(n_estimators = k, algorithm = j)
                    model_k.fit(X_training, y_training)

                    # compute the predictions for the training and test sets
                    predictions_training_k = model_k.predict(X_training)
                    predictions_test_k = model_k.predict(X_test)

                    # compute the accuracy on the training and test set predictions
                    accuracy_training_k.append(metrics.accuracy_score(y_training, predictions_training_k))
                    accuracy_test_k.append(metrics.accuracy_score(y_test, predictions_test_k))

            print(accuracy_training_k)
            print(accuracy_test_k)

            print("Average Accuracy for Training: " + str(mean(accuracy_training_k)))
            print("Average Accuracy for Test: " + str(mean(accuracy_test_k)))

            # let's plot the accuracy on the training and test set
            plt.scatter(k_values,accuracy_training_k,marker="x")
            plt.scatter(k_values,accuracy_test_k,marker="+")
            plt.xlim([0, max(k_values)+2])
            plt.ylim([0.0, 1.1])
            plt.xlabel("Value of k (n_estimators) with j (algorithm): " + str(j))
            plt.ylabel("Accuracy")
            legend_labels = ["Training (Euclidian dist.)","Test (Euclidian dist.)"]
            plt.legend(labels=legend_labels, loc=4, borderpad=1)
            plt.title("Effect of k and j on training and test set accuracy", fontsize=10)
            plt.show()

            # Empty for next iteration
            accuracy_training_k = []
            accuracy_test_k = []