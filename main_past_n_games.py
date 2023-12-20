import datetime
import pandas as pd
import matplotlib.pyplot as plt
import sys

from load_dataset import load_dataset
from find_related_games import find_any_past_n_games

n_games = 50 # number of past games to consider
limit = None # truncate dataframe for testing (saves time) - None for complete df

df = load_dataset("local/games.csv") # load dataset from csv into dataframe

# df.boxplot(["FG_PCT_home",
#             "FT_PCT_home",
#             "FG3_PCT_home",
#             "FG_PCT_away",
#             "FT_PCT_away",
#             "FG3_PCT_away",
#             ])
#
# plt.show()
#
# df.boxplot(["PTS_home",
#             "AST_home",
#             "REB_home",
#             "PTS_away",
#             "AST_away",
#             "REB_away"
#             ])
#
# plt.show()

# remove outliers
df.drop(df[df.FG3_PCT_home > .90].index, inplace = True)
df.drop(df[df.FG3_PCT_away > .90].index, inplace = True)
df.drop(df[df.PTS_home < 50].index, inplace = True)
df.drop(df[df.PTS_away < 50].index, inplace = True)
df.drop(df[df.REB_away > 75].index, inplace = True)

# df.boxplot(["FG_PCT_home",
#             "FT_PCT_home",
#             "FG3_PCT_home",
#             "FG_PCT_away",
#             "FT_PCT_away",
#             "FG3_PCT_away",
#             ])
#
# plt.show()
#
# df.boxplot(["PTS_home",
#             "AST_home",
#             "REB_home",
#             "PTS_away",
#             "AST_away",
#             "REB_away"
#             ])
#
# plt.show()

df_aug = find_any_past_n_games(df, n_games, limit)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# create working copy
#df_work = df_aug.copy()

# scale values before
x = df_aug.values

# scaler = preprocessing.StandardScaler().fit(x)
# x_scaled = scaler.transform(x)

scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)

df_work = pd.DataFrame(x_scaled, columns = df_aug.columns, index = df_aug.index)
df_work["HOME_TEAM_WINS"] = df_work["HOME_TEAM_WINS"].astype(dtype=int)

Y = df_work["HOME_TEAM_WINS"]
X = df_work.drop("HOME_TEAM_WINS", axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=26)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, Y_train)
Y_rf_class = rf_classifier.predict(X_test)
rf_class_accuracy = accuracy_score(Y_rf_class, Y_test)

print(rf_class_accuracy)
