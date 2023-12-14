from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def random_forest_learning(df):
    """Learning of random forest classifier"""

    df_work = df.convert_dtypes().copy()

    y = df_work["HOME_TEAM_WINS"]
    X = df_work.drop("HOME_TEAM_WINS", axis=1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)

    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    y_rf_class = rf_classifier.predict(X_test)
    rf_class_accuracy = accuracy_score(y_rf_class, y_test)

    print(f'RANDOM FOREST CLASS ACCURACY {rf_class_accuracy}')
    print('RF complete')

    return rf_classifier
    # return value does not make any sense without 
    # the split up dataset