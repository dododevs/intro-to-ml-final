import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from load_dataset import load_dataset
from find_related_games import find_any_past_n_games
from learning import random_forest_learning

n_games = 4 # number of past games to consider
limit = None # truncate dataframe for testing (saves time) - None for complete df
sample_size = 100   # number of times to train model to get good accuracy mean

df = load_dataset("local/games.csv") # load dataset from csv into dataframe

df_aug = find_any_past_n_games(df, n_games, limit)

## acutall learning
# define params
X_labels = df_aug.keys().values[1:]
y_label = "HOME_TEAM_WINS"
scaler = "MinMax"
# learn model
rf_class_accuracy = random_forest_learning(df_aug, X_labels, y_label, scaler, random_state = 26)

# learn model multiple times for statistical analysis of results
store = np.empty(sample_size)
for i in range(sample_size):
    print(f'Learning model {i}/{sample_size}')
    store[i] = random_forest_learning(df_aug, X_labels, y_label, scaler)
print(store)
print(f'Mean: {np.mean(store)}')
print(f'std : {np.std(store)}')
print(f'min : {np.min(store)}')
print(f'max : {np.max(store)}')





# just not to forget the syntax xD
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
#df.drop(df[df.FG3_PCT_home > .90].index, inplace = True)
#df.drop(df[df.FG3_PCT_away > .90].index, inplace = True)
#df.drop(df[df.PTS_home < 50].index, inplace = True)
#df.drop(df[df.PTS_away < 50].index, inplace = True)
#df.drop(df[df.REB_away > 75].index, inplace = True)

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
