import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def PlotGraph(j_values, training_R_values, training_MAE_values, test_R_values, test_MAE_values):

    plt.scatter(j_values, training_R_values[0], marker="x")
    plt.scatter(j_values, test_R_values[0], marker="+")
    plt.scatter(j_values, training_MAE_values[0], marker="^")
    plt.scatter(j_values, test_MAE_values[0], marker="v")

    plt.scatter(j_values, training_R_values[1], marker="p")
    plt.scatter(j_values, test_R_values[1], marker="*")
    plt.scatter(j_values, training_MAE_values[1], marker="D")
    plt.scatter(j_values, test_MAE_values[1], marker="d")

    plt.xlim([0, max(j_values)+2])
    plt.ylim([-100, 100])
    plt.xlabel("Value of j")
    plt.ylabel("Accuracy score based on mean, std, MAE")
    legend_labels = ["Training mean R value", "Test mean R value", "Training mean MAE value", "Test mean MAE value", "Training std R value", "Test std R value", "Training std MAE value", "Test std MAE value"]
    plt.legend(labels=legend_labels, loc=4, borderpad=1)
    plt.title("Effect of j on training and test set accuracy (mean, std, MAE)", fontsize=10)
    plt.show()



    
    
def DecisionTreeRegressionFunction(X,y):

    j_values = list(range(1,50,5))
    k_values = ["auto", "sqrt", "log2"]

    DecisionTreeRegressor = tree.DecisionTreeRegressor()
    parameters = {'max_features': (*k_values,), 'max_depth': j_values}

    # Mae is domain dependent
    # R is domain independent
    
    clf = GridSearchCV(DecisionTreeRegressor, parameters, cv=10, refit="R", return_train_score=True, scoring={"MAE":"neg_mean_absolute_error", "R": "r2"})

    # Fit the regressor model
    clf.fit(X,y)

    training_mean_R_value = np.round((clf.cv_results_['mean_train_R'])[:10], 3)
    training_std_R_value = np.round((clf.cv_results_['std_train_R'])[:10], 3)
   
    test_mean_R_value = np.round((clf.cv_results_['mean_test_R'])[:10], 3)
    test_std_R_value = np.round((clf.cv_results_['std_test_R'])[:10], 3)

    training_mean_MAE_value = np.round((clf.cv_results_['mean_train_MAE'])[:10],3)
    training_std_MAE_value = np.round((clf.cv_results_['std_train_MAE'])[:10],3)

    test_mean_MAE_value = np.round((clf.cv_results_['mean_test_MAE'])[:10],3)
    test_std_MAE_value = np.round((clf.cv_results_['std_test_MAE'])[:10],3)

    
    print(f"\nBest Parameters for Decision tree Regressor:\n {clf.best_params_}\n")
    print(f"Best score for Decision tree Regressor:\n {clf.best_score_:.3f}\n")

    print(f"Mean r2 for Decision tree Regressor (training):\n {training_mean_R_value}\n")
    print(f"Std r2 for Decision tree Regressor (training):\n {training_std_R_value}\n")

    print(f"Mean r2 for Decision tree Regressor (test):\n {test_mean_R_value}\n")
    print(f"Std r2 for Decision tree Regressor (test):\n {test_std_R_value}\n")

    print(f"Mean MAE for Decision tree Regressor (training):\n {training_mean_MAE_value}\n")
    print(f"Std MAE for Decision tree Regressor (training):\n {training_std_MAE_value}\n")

    print(f"Mean MAE for Decision tree Regressor (test):\n {test_mean_MAE_value}\n")
    print(f"Std MAE for Decision tree Regressor (test):\n {test_std_MAE_value}\n")


    training_R_values = [training_mean_R_value, training_std_R_value]
    training_MAE_values = [training_mean_MAE_value, training_std_MAE_value]

    test_R_values = [test_mean_R_value, test_std_R_value]
    test_MAE_values = [test_mean_MAE_value, test_std_MAE_value]

    
    PlotGraph(j_values, training_R_values, training_MAE_values, test_R_values, test_MAE_values)


    
    







        
        


      






