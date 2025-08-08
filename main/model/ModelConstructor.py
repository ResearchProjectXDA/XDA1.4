# Import for Construct Defect Models (Classification)
from enum import unique
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.ensemble import RandomForestClassifier  # Random Forests
from sklearn.tree import DecisionTreeClassifier  # C5.0 (Decision Tree)
from sklearn.neural_network import MLPClassifier  # Neural Network
from sklearn.ensemble import GradientBoostingClassifier  # Gradient Boosting Machine (GBM)
import xgboost as xgb  # eXtreme Gradient Boosting Tree (xGBTree)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def constructModel(X_train, X_test, y_train, y_test, export=False):

    #Use a loss function that accounts for class imbalance
    unique, counts = np.unique(y_train, return_counts=True)
    counter = dict(zip(unique, counts))
    n_labels = len(y_train)

    # The class weights are calculated as the ratio of the total number of labels to the count of each class, this way the less frequent class gets a higher weight
    class_weights = {cls: n_labels / count for cls, count in counter.items()}
    print("Class weights: ", class_weights)
    print("Counter: ", counter)

    # Logistic Regression
    lr_model = LogisticRegression(random_state=1234, class_weight=class_weights)
    lr_model.fit(X_train, y_train)
    lr_model_AUC = round(roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1]), 3)
    y_pred = lr_model.predict(X_test)
    lr_model_f1 = round(f1_score(y_test, y_pred), 3)
    print('Logistic Regression F1 score: ' + str(lr_model_f1))
    lr_model_accuracy = round(accuracy_score(y_test, y_pred), 3)
    print('Logistic Regression Accuracy score: ' + str(lr_model_accuracy))
    lr_model_recall = round(recall_score(y_test, y_pred), 3)
    print('Logistic Regression Recall score: ' + str(lr_model_recall))
    lr_model_precision = round(precision_score(y_test, y_pred), 3)
    print('Logistic Regression Precision score: ' + str(lr_model_precision))

    # Random Forests
    rf_model = RandomForestClassifier(random_state=1234, n_jobs=10, class_weight=class_weights)
    rf_model.fit(X_train, y_train)
    rf_model_AUC = round(roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]), 3)
    rf_model_f1 = round(f1_score(y_test, rf_model.predict(X_test)), 3)
    print('Random Forests F1 score: ' + str(rf_model_f1))
    rf_model_accuracy = round(accuracy_score(y_test, rf_model.predict(X_test)), 3)
    print('Random Forests Accuracy score: ' + str(rf_model_accuracy))
    rf_model_recall = round(recall_score(y_test, rf_model.predict(X_test)), 3)
    print('Random Forests Recall score: ' + str(rf_model_recall))
    rf_model_precision = round(precision_score(y_test, rf_model.predict(X_test)), 3)
    print('Random Forests Precision score: ' + str(rf_model_precision))

    # C5.0 (Decision Tree)
    dt_model = DecisionTreeClassifier(random_state=1234, class_weight=class_weights)
    dt_model.fit(X_train, y_train)
    dt_model_AUC = round(roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1]), 3)
    dt_model_f1 = round(f1_score(y_test, dt_model.predict(X_test)), 3)
    print('C5.0 (Decision Tree) F1 score: ' + str(dt_model_f1))
    dt_model_accuracy = round(accuracy_score(y_test, dt_model.predict(X_test)), 3)
    print('C5.0 (Decision Tree) Accuracy score: ' + str(dt_model_accuracy))
    dt_model_recall = round(recall_score(y_test, dt_model.predict(X_test)), 3)
    print('C5.0 (Decision Tree) Recall score: ' + str(dt_model_recall))
    dt_model_precision = round(precision_score(y_test, dt_model.predict(X_test)), 3)
    print('C5.0 (Decision Tree) Precision score: ' + str(dt_model_precision))

    # Neural Network
    # no class_weight and sample_weight for MLPClassifier as it does not support them
    nn_model = MLPClassifier(random_state=1234)
    nn_model.fit(X_train, y_train)
    nn_model_AUC = round(roc_auc_score(y_test, nn_model.predict_proba(X_test)[:, 1]), 3)
    nn_model_f1 = round(f1_score(y_test, nn_model.predict(X_test)), 3)
    print('Neural Network F1 score: ' + str(nn_model_f1))
    nn_model_accuracy = round(accuracy_score(y_test, nn_model.predict(X_test)), 3)
    print('Neural Network Accuracy score: ' + str(nn_model_accuracy))
    nn_model_recall = round(recall_score(y_test, nn_model.predict(X_test)), 3)
    print('Neural Network Recall score: ' + str(nn_model_recall))
    nn_model_precision = round(precision_score(y_test, nn_model.predict(X_test)), 3)
    print('Neural Network Precision score: ' + str(nn_model_precision))

    # Gradient Boosting Machine (GBM)
    # Use sample weights to handle class imbalance since GradientBoostingClassifier does not support class_weight
    sample_weights = np.array([class_weights[y] for y in y_train])

    gbm_model = GradientBoostingClassifier(random_state=1234)
    gbm_model.fit(X_train, y_train, sample_weight=sample_weights)

    gbm_model_AUC = round(roc_auc_score(y_test, gbm_model.predict_proba(X_test)[:, 1]), 3)
    gbm_model_f1 = round(f1_score(y_test, gbm_model.predict(X_test)), 3)
    print('Gradient Boosting Machine (GBM) F1 score: ' + str(gbm_model_f1))
    gbm_model_accuracy = round(accuracy_score(y_test, gbm_model.predict(X_test)), 3)
    print('Gradient Boosting Machine (GBM) Accuracy score: ' + str(gbm_model_accuracy))
    gbm_model_recall = round(recall_score(y_test, gbm_model.predict(X_test)), 3)
    print('Gradient Boosting Machine (GBM) Recall score: ' + str(gbm_model_recall))
    gbm_model_precision = round(precision_score(y_test, gbm_model.predict(X_test)), 3)
    print('Gradient Boosting Machine (GBM) Precision score: ' + str(gbm_model_precision))

    # eXtreme Gradient Boosting Tree (xGBTree)
    # Use scale_pos_weight to handle class imbalance since xgboost does not support class_weight nor sample_weight
    scale_pos_weight = counter[0] / counter[1]

    xgb_model = xgb.XGBClassifier(random_state=1234, scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_model_AUC = round(roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]), 3)
    xgb_model_f1 = round(f1_score(y_test, xgb_model.predict(X_test)), 3)
    print('eXtreme Gradient Boosting Tree (xGBTree) F1 score: ' + str(xgb_model_f1))
    xgb_model_accuracy = round(accuracy_score(y_test, xgb_model.predict(X_test)), 3)
    print('eXtreme Gradient Boosting Tree (xGBTree) Accuracy score: ' + str(xgb_model_accuracy))
    xgb_model_recall = round(recall_score(y_test, xgb_model.predict(X_test)), 3)
    print('eXtreme Gradient Boosting Tree (xGBTree) Recall score: ' + str(xgb_model_recall))
    xgb_model_precision = round(precision_score(y_test, xgb_model.predict(X_test)), 3)
    print('eXtreme Gradient Boosting Tree (xGBTree) Precision score: ' + str(xgb_model_precision))

    models = {
        'Logistic Regression': lr_model,
        'Random Forests': rf_model,
        'C5.0 (Decision Tree)': dt_model,
        'Neural Network': nn_model,
        'Gradient Boosting Machine (GBM)': gbm_model,
        'eXtreme Gradient Boosting Tree (xGBTree)': xgb_model,
    }

    # Summarise into a DataFrame using F1 scores
    model_performance_df = pd.DataFrame(data=np.array([list(models.keys()),
                [lr_model_f1, rf_model_f1, dt_model_f1, nn_model_f1, gbm_model_f1, xgb_model_f1]]).transpose(),
                index=range(6),
                columns=['Model', 'F1'])
    model_performance_df['F1'] = model_performance_df.F1.astype(float)
    model_performance_df = model_performance_df.sort_values(by=['F1'], ascending=False)

    bestModel = model_performance_df.iloc[0].iloc[0]

    print(model_performance_df)
    print('Best model is: ' + bestModel)

    # Visualise the performance of defect models
    if export:
        display(model_performance_df)
        model_performance_df.plot(kind='barh', y='F1', x='Model')
        plt.tight_layout()
        plt.savefig('../plots/F1.png')
        plt.show()

    return models[bestModel]
