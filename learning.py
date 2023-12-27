from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def random_forest_learning(df, X_labels, y_label, scaler=None, test_size=0.2, random_state=None):
    """Learning of random forest classifier"""
    # scaler can be one of the strings "MinMax" or "Standard" or None for no scaling

    df_work = df.convert_dtypes().copy()

    y = df_work[y_label]
    X = df_work[X_labels]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    match scaler:
        case "MinMax":
            print('Performing MinMax scaling on data')
            scl = MinMaxScaler()
            X_train = scl.fit_transform(X_train)
            X_test = scl.transform(X_test)
        case "Standard":
            print('Performing Standard scaling on data')
            scl = StandardScaler()
            X_train = scl.fit_transform(X_train)
            X_test = scl.transform(X_test)
        case _:
            print('No scaling')

    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    y_rf_class = rf_classifier.predict(X_test)
    rf_class_accuracy = accuracy_score(y_rf_class, y_test)

    print(f'RANDOM FOREST CLASS ACCURACY {rf_class_accuracy}')
    print('RF complete')

    # return accuracy for statistical analysis
    return rf_class_accuracy

    # learner as return value does not make any sense without the split up dataset
    # return rf_classifier
