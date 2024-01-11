from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, f1_score, precision_recall_curve, roc_curve, auc
import numpy as np

from timeit import default_timer as timer

def random_forest_learning(df, X_labels, y_label, scaler=None, test_size=0.8, random_state=None, min_samples_split=2, max_depth=None):
    return classification_learning("rf", df, X_labels, y_label, scaler, test_size, random_state, min_samples_split, max_depth)

def support_vector_learning(df, X_labels, y_label, scaler=None, test_size=0.8, random_state=None):
    return classification_learning("svc", df, X_labels, y_label, scaler, test_size, random_state)

def bayes_learning(df, X_labels, y_label, scaler=None, test_size=0.8, random_state=None):
    return classification_learning("nb", df, X_labels, y_label, scaler, test_size, random_state)

def knn_learning(df, X_labels, y_label, scaler=None, test_size=0.8, random_state=None):
    return classification_learning("knn", df, X_labels, y_label, scaler, test_size, random_state)

def dummy_learning(df, X_labels, y_label, scaler=None, test_size=0.8, random_state=None):
    return classification_learning("dummy", df, X_labels, y_label, scaler, test_size, random_state)

def classification_learning(model, df, X_labels, y_label, scaler, test_size, random_state, min_samples_split=2, max_depth=None):
    df_work = df.convert_dtypes().copy()

    y = df_work[y_label]
    X = df_work[X_labels]

    # perform grid search only on small part of the data (which is still a about 5000 rows for past 37 games)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # define classifier and grid parameters
    match model:
        case "rf":
            print('Start Random Forest classification')
            classifier = RandomForestClassifier()
            param_grid = dict(clf__min_samples_split = range(2,8),
                              clf__max_depth = range(3,13))
        case "svc":
            print('Start Support Vector classification')
            classifier = SVC(probability=True, cache_size=2000)
            param_grid = dict(clf__gamma = [2**p for p in range(-15,4)], clf__C = [2**p for p in range(-5, 16)])
        case "nb":
            print('Start Naive Bayes classification')
            classifier = BernoulliNB(force_alpha = True)
            param_grid = dict(clf__alpha = np.linspace(1e-10,1.4,50))
        case "knn":
            print('Start k-nearest neighbors classification')
            classifier = KNeighborsClassifier()
            param_grid = dict(clf__n_neighbors = range(3,12,2),
                              clf__weights = ["uniform", "distance"],
                              clf__p = [1,2])
        case "dummy":
            print('Start dummy classification')
            classifier = DummyClassifier()
            param_grid = dict()
        case _:
            raise Exception('Unknown classification model')

    # define scaler
    match scaler:
        case "MinMax":
            scl = MinMaxScaler()
            print('MinMax scaling done')
        case "Standard":
            scl = StandardScaler()
            print('Standard scaling done')
        case _:
            print('No scaling')

    # pipeline glues scaler and classiefier together
    pipe = Pipeline([
            ('scale', scl),
            ('clf', classifier)])

    # automatic grid seach on afore defined params per classifier
    gsc = GridSearchCV(pipe, param_grid = param_grid, scoring='neg_mean_squared_error', n_jobs=-1, refit=True, verbose = 10)

    start = timer()
    gsc.fit(X_train, y_train) # start actual grid search
    y_class = gsc.predict(X_test)
    time_consumption = timer() - start

    # calculate more assessment scores
    accuracy = accuracy_score(y_class, y_test)
    probs = gsc.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    if model == "rf":
        # calculate feature importances
        importances = gsc.best_estimator_.named_steps['clf'].feature_importances_
        indices = np.argsort(importances)[::-1]
        print(f"Feature ranking for {model}:")
        for f in range(X_train.shape[1]):
            print(f"{f + 1:>2}. Feature {X_labels[indices[f]]:<17} - importance: {importances[indices[f]]:.6f}")

    print(f"Best params are {gsc.best_params_}")
    print("Classification training and assessment computation done\n")

    return accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption, gsc.best_params_

# old code not used anymore
def classification_learning_old(model, df, X_labels, y_label, scaler, test_size, random_state, min_samples_split=2, max_depth=None):
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
            classifier = RandomForestClassifier(min_samples_split=min_samples_split, max_depth=max_depth, n_jobs=4)
        case "svc":
            print('Start Support Vector classification')
            classifier = SVC(probability=True, kernel='poly', cache_size=2000)
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
            print(f"{f + 1:>2}. Feature {X_labels[indices[f]]:<17} - importance: {importances[indices[f]]}")

        # Access a specific tree from the forest (in this case, the first tree, change index as needed)
        # tree_to_inspect = classifier.estimators_[0]

        # Get information about the tree
        # print(f"Tree Information:")
        # print(f"Number of nodes: {tree_to_inspect.tree_.node_count}")
        # print(f"Tree depth: {tree_to_inspect.tree_.max_depth}")
        # print(f"Feature importance: {tree_to_inspect.feature_importances_}")
        # print("-----------------------")

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
