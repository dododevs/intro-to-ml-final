from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, f1_score, precision_recall_curve, roc_curve, auc
import numpy as np

from timeit import default_timer as timer

def random_forest_learning(df, X_labels, y_label, scaler=None, test_size=0.2, random_state=None):
    return classification_learning("rf", df, X_labels, y_label, scaler, test_size, random_state)

def support_vector_learning(df, X_labels, y_label, scaler=None, test_size=0.2, random_state=None):
    return classification_learning("svc", df, X_labels, y_label, scaler, test_size, random_state)

def bayes_learning(df, X_labels, y_label, scaler=None, test_size=0.2, random_state=None):
    return classification_learning("nb", df, X_labels, y_label, scaler, test_size, random_state)

def knn_learning(df, X_labels, y_label, scaler=None, test_size=0.2, random_state=None):
    return classification_learning("knn", df, X_labels, y_label, scaler, test_size, random_state)

def dummy_learning(df, X_labels, y_label, scaler=None, test_size=0.2, random_state=None):
    return classification_learning("dummy", df, X_labels, y_label, scaler, test_size, random_state)

def classification_learning(model, df, X_labels, y_label, scaler, test_size, random_state):
    """Learning of classifier"""
    # scaler can be one of the strings "MinMax" or "Standard" or None for no scaling

    df_work = df.convert_dtypes().copy()

    y = df_work[y_label]
    X = df_work[X_labels]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train, X_test = scale_train_test(X_train, X_test, scaler)

    match model:
        case "rf":
            print('Start Random Forest classification')
            classifier = RandomForestClassifier()# min_samples_split=1000, max_depth=4)
        case "svc":
            print('Start Support Vector classification')
            classifier = SVC(probability=True)
        case "nb":
            print('Start Naive Bayes classification')
            classifier = BernoulliNB()
        case "knn":
            print('Start k-nearest neighbors classification')
            classifier = KNeighborsClassifier()
        case "dummy":
            print('Start dummy classification')
            classifier = DummyClassifier()
        case _:
            raise Exception('Unknown classification model')
    # train the classifier, meassure wall time consumption of training plus
    # simple (non-probabilistic) prediction as simple efficency score
    start = timer()
    classifier.fit(X_train, y_train)
    y_class = classifier.predict(X_test)
    time_consumption = timer() - start

    # calculate more assessment scores
    accuracy = accuracy_score(y_class, y_test)
    probs = classifier.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    # print additional information if the model is random forest:
    if model == "rf":
        # calculate feature importances
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        print(f"Feature ranking for {model}:")
        for f in range(X_train.shape[1]):
            print(f"{f + 1}. Feature {X_labels[indices[f]]} - importance: {importances[indices[f]]}")

        # Access a specific tree from the forest (in this case, the first tree, change index as needed)
        tree_to_inspect = classifier.estimators_[0]

        # Get information about the tree
        print(f"Tree Information:")
        print(f"Number of nodes: {tree_to_inspect.tree_.node_count}")
        print(f"Tree depth: {tree_to_inspect.tree_.max_depth}")
        print(f"Feature importance: {tree_to_inspect.feature_importances_}")
        print("-----------------------")

    print("Classification training and assessment computation done\n")

    # return assessment scores
    return accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption

def scale_train_test(X_train, X_test, scaler):
    match scaler:
        case "MinMax":
            scl = MinMaxScaler()
            X_train = scl.fit_transform(X_train)
            X_test = scl.transform(X_test)
            print('MinMax scaling done')
        case "Standard":
            scl = StandardScaler()
            X_train = scl.fit_transform(X_train)
            X_test = scl.transform(X_test)
            print('Standard scaling done')
        case _:
            print('No scaling')
    return X_train, X_test
