import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain, combinations

# user functions
from load_dataset import load_dataset
from find_related_games import find_any_past_n_games
from learning import random_forest_learning, support_vector_learning, bayes_learning, knn_learning, dummy_learning
from plot import compare_binary_classification

import sys
import os

#n_games = int(sys.argv[1]) # number of past games to consider - used command line args to write bash script iteration...
n_games = 48
limit = None # truncate dataframe for testing (saves time) - None for complete df

df = load_dataset("local/games.csv") # load dataset from csv into dataframe

df.drop(df[df.GAME_DATE_EST < datetime.datetime(2019, 6, 1)].index, inplace=True)

df_aug = find_any_past_n_games(df, n_games, limit) # find the past stats to every game

# calc mean of previous stats
df_aug["prev_PTS_home"] = df_aug[[s for s in df_aug.columns if 'prev_PTS_home_' in s]].sum(axis=1) / n_games
df_aug["prev_FG_PCT_home"] = df_aug[[s for s in df_aug.columns if 'prev_FG_PCT_home_' in s]].sum(axis=1) / n_games
df_aug["prev_FT_PCT_home"] = df_aug[[s for s in df_aug.columns if 'prev_FT_PCT_home_' in s]].sum(axis=1) / n_games
df_aug["prev_FG3_PCT_home"] = df_aug[[s for s in df_aug.columns if 'prev_FG3_PCT_home_' in s]].sum(axis=1) / n_games
df_aug["prev_AST_home"] = df_aug[[s for s in df_aug.columns if 'prev_AST_home_' in s]].sum(axis=1) / n_games
df_aug["prev_REB_home"] = df_aug[[s for s in df_aug.columns if 'prev_REB_home_' in s]].sum(axis=1) / n_games

df_aug["prev_PTS_away"] = df_aug[[s for s in df_aug.columns if 'prev_PTS_away_' in s]].sum(axis=1) / n_games
df_aug["prev_FG_PCT_away"] = df_aug[[s for s in df_aug.columns if 'prev_FG_PCT_away_' in s]].sum(axis=1) / n_games
df_aug["prev_FT_PCT_away"] = df_aug[[s for s in df_aug.columns if 'prev_FT_PCT_away_' in s]].sum(axis=1) / n_games
df_aug["prev_FG3_PCT_away"] = df_aug[[s for s in df_aug.columns if 'prev_FG3_PCT_away_' in s]].sum(axis=1) / n_games
df_aug["prev_AST_away"] = df_aug[[s for s in df_aug.columns if 'prev_AST_away_' in s]].sum(axis=1) / n_games
df_aug["prev_REB_away"] = df_aug[[s for s in df_aug.columns if 'prev_REB_away_' in s]].sum(axis=1) / n_games

# calculate differences in stats
df_aug["prev_PTS_diff"] = df_aug["prev_PTS_home"] - df_aug["prev_PTS_away"]
df_aug["prev_FG_PCT_diff"] = df_aug["prev_FG_PCT_home"] - df_aug["prev_FG_PCT_away"]
df_aug["prev_FT_PCT_diff"] = df_aug["prev_FT_PCT_home"] - df_aug["prev_FT_PCT_away"]
df_aug["prev_FG3_PCT_diff"] = df_aug["prev_FG3_PCT_home"] - df_aug["prev_FG3_PCT_away"]
df_aug["prev_AST_diff"] = df_aug["prev_AST_home"] - df_aug["prev_AST_away"]
df_aug["prev_REB_diff"] = df_aug["prev_REB_home"] - df_aug["prev_REB_away"]

#drop previous stats
end_index = n_games * 12
df_aug.drop(df_aug.keys().values[1:end_index+1], axis = 1, inplace = True)

df_work = df_aug.copy()

## acutall learning
# define params
X_labels = df_work.keys().values[1:]
y_label = "HOME_TEAM_WINS"
scaler = "MinMax" # scaling used in preprocessing MinMax or Standard

# prepare containers for assessment ploting
model_name_plot, prec_rec_plot, pr_auc_plot, fpr_tpr_plot, roc_auc_plot, acc_plot, time_plot  = [[] for _ in range(7)]

# learn random forest
accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption, params = random_forest_learning(df_work, X_labels, y_label, scaler=scaler, random_state = 26)

# store assessment scores
model_name_plot.append(f'RF')
acc_plot.append(accuracy)
prec_rec_plot.append([precision, recall])
pr_auc_plot.append(pr_auc)
fpr_tpr_plot.append([fpr, tpr])
roc_auc_plot.append(roc_auc)
time_plot.append(time_consumption)

# learn support vector machine
accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption, params = support_vector_learning(df_work, X_labels, y_label, scaler, random_state = 26)

# store assessment scores
model_name_plot.append('SVC')
acc_plot.append(accuracy)
prec_rec_plot.append([precision, recall])
pr_auc_plot.append(pr_auc)
fpr_tpr_plot.append([fpr, tpr])
roc_auc_plot.append(roc_auc)
time_plot.append(time_consumption)

# learn naive bayes
accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption, params = bayes_learning(df_work, X_labels, y_label, scaler, random_state = 26)

# store assessment scores
model_name_plot.append('NB')
acc_plot.append(accuracy)
prec_rec_plot.append([precision, recall])
pr_auc_plot.append(pr_auc)
fpr_tpr_plot.append([fpr, tpr])
roc_auc_plot.append(roc_auc)
time_plot.append(time_consumption)

# learn k-nearest neighbors
accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption, params = knn_learning(df_work, X_labels, y_label, scaler, random_state = 26)

# store assessment scores
model_name_plot.append('KNN')
acc_plot.append(accuracy)
prec_rec_plot.append([precision, recall])
pr_auc_plot.append(pr_auc)
fpr_tpr_plot.append([fpr, tpr])
roc_auc_plot.append(roc_auc)
time_plot.append(time_consumption)

# learn dummy
accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption, params = dummy_learning(df_work, X_labels, y_label, scaler, random_state = 26)

# store assessment scores
model_name_plot.append('DUMMY')
acc_plot.append(accuracy)
prec_rec_plot.append([precision, recall])
pr_auc_plot.append(pr_auc)
fpr_tpr_plot.append([fpr, tpr])
roc_auc_plot.append(roc_auc)
time_plot.append(time_consumption)

compare_binary_classification(model_name_plot, prec_rec_plot, pr_auc_plot, fpr_tpr_plot, roc_auc_plot, acc_plot, time_plot)
