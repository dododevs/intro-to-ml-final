import datetime
import numpy as np
import pandas as pd

# user functions
from load_dataset import load_dataset
from find_related_games import find_any_past_n_games
from learning import random_forest_learning, support_vector_learning, bayes_learning, knn_learning, dummy_learning
from plot import compare_binary_classification

import sys

n_games = int(sys.argv[1]) # number of past games to consider
limit = None # truncate dataframe for testing (saves time) - None for complete df
sample_size = 100   # number of times to train model to get good accuracy mean
min_samples_split = [2] # This parameter does not seem to make a difference
max_depth = list(range(4,5,1))         # change max_depth parameter here

df = load_dataset("local/games.csv") # load dataset from csv into dataframe

# calculate differences in stats --- work in progress, needs also adjustments ind find_any_past_n_games and maybe further down this script
# df_aug["PTS_diff"] = df["PTS_home"] - df["PTS_away"]
# df_aug["FG_PCT_diff"] = df_aug[[s for s in df_aug.columns if 'prev_FG_PCT_home_' in s]].sum(axis=1) / n_games
# df_aug["FT_PCT_diff"] = df_aug[[s for s in df_aug.columns if 'prev_FT_PCT_home_' in s]].sum(axis=1) / n_games
# df_aug["FG3_PCT_diff"] = df_aug[[s for s in df_aug.columns if 'prev_FG3_PCT_home_' in s]].sum(axis=1) / n_games
# df_aug["AST_diff"] = df_aug[[s for s in df_aug.columns if 'prev_AST_home_' in s]].sum(axis=1) / n_games
# df_aug["REB_diff"] = df_aug[[s for s in df_aug.columns if 'prev_REB_home_' in s]].sum(axis=1) / n_games

df_aug = find_any_past_n_games(df, n_games, limit)

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

#drop previous stats
end_index = n_games * 12
df_aug.drop(df_aug.keys().values[1:end_index+1], axis = 1, inplace = True)
# drop features with little importance
df_aug.drop(["prev_FT_PCT_away", "prev_FT_PCT_home"], axis = 1, inplace = True)
#df_aug.drop(["prev_AST_home", "prev_AST_away"], axis = 1, inplace = True) # tests show, that the existence of AST is marginally better
df_aug.info()

# prepare containers for assessment ploting
model_name_plot, prec_rec_plot, pr_auc_plot, fpr_tpr_plot, roc_auc_plot, acc_plot, time_plot  = [[] for _ in range(7)]

## acutall learning
# define params
X_labels = df_aug.keys().values[1:]
y_label = "HOME_TEAM_WINS"
scaler = "MinMax"

print(f'Iterating through RF parameters:\n  min_samples_split:{min_samples_split}\n  max_depth:{max_depth}')

# for mins in min_samples_split:
#     for maxd in max_depth:
#         print(f'min_samples_split: {mins} max_depth: {maxd}')
#
#         # learn random forest
#         accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption = random_forest_learning(df_aug, X_labels, y_label, scaler, random_state = 26, min_samples_split=mins, max_depth=maxd)
#
#         # store assessment scores
#         model_name_plot.append(f'RF{maxd}')
#         acc_plot.append(accuracy)
#         prec_rec_plot.append([precision, recall])
#         pr_auc_plot.append(pr_auc)
#         fpr_tpr_plot.append([fpr, tpr])
#         roc_auc_plot.append(roc_auc)
#         time_plot.append(time_consumption)
#
# # learn support vector machine
# accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption = support_vector_learning(df_aug, X_labels, y_label, scaler, random_state = 26)
#
# # store assessment scores
# model_name_plot.append('SVC')
# acc_plot.append(accuracy)
# prec_rec_plot.append([precision, recall])
# pr_auc_plot.append(pr_auc)
# fpr_tpr_plot.append([fpr, tpr])
# roc_auc_plot.append(roc_auc)
# time_plot.append(time_consumption)
#
# # learn naive bayes
# accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption = bayes_learning(df_aug, X_labels, y_label, scaler, random_state = 26)
#
# # store assessment scores
# model_name_plot.append('NB')
# acc_plot.append(accuracy)
# prec_rec_plot.append([precision, recall])
# pr_auc_plot.append(pr_auc)
# fpr_tpr_plot.append([fpr, tpr])
# roc_auc_plot.append(roc_auc)
# time_plot.append(time_consumption)
#
# # learn k-nearest neighbors
# accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption = knn_learning(df_aug, X_labels, y_label, scaler, random_state = 26)
#
# # store assessment scores
# model_name_plot.append('KNN')
# acc_plot.append(accuracy)
# prec_rec_plot.append([precision, recall])
# pr_auc_plot.append(pr_auc)
# fpr_tpr_plot.append([fpr, tpr])
# roc_auc_plot.append(roc_auc)
# time_plot.append(time_consumption)
#
# # learn dummy
# accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption = dummy_learning(df_aug, X_labels, y_label, scaler, random_state = 26)
#
# # store rf assessment scores
# model_name_plot.append('DUMMY')
# acc_plot.append(accuracy)
# prec_rec_plot.append([precision, recall])
# pr_auc_plot.append(pr_auc)
# fpr_tpr_plot.append([fpr, tpr])
# roc_auc_plot.append(roc_auc)
# time_plot.append(time_consumption)
#
# compare_binary_classification(model_name_plot, prec_rec_plot, pr_auc_plot, fpr_tpr_plot, roc_auc_plot, acc_plot, time_plot)

# learn model multiple times for statistical analysis of results
if sample_size > 0:
    store = np.empty([len(max_depth), sample_size])
    for i in range(len(max_depth)):
        for j in range(sample_size):
            print(f'Learning model {j+1}/{sample_size} with max_depth = {max_depth[i]}/{max_depth}')
            accuracy, _, _, _, _, _, _, _ = random_forest_learning(df_aug, X_labels, y_label, scaler, random_state = 26, max_depth=max_depth[i])
            store[i][j] = accuracy

    max_acci = [0, 0] # first entry is index and second is maximum of accuracies
    for i in range(len(max_depth)):
        acc_mean = np.mean(store[i])
        if acc_mean > max_acci[0]:
            max_acci = [i, acc_mean]
    print(f'Maximum accuracy is {max_acci[1]} with max_depth = {max_depth[max_acci[0]]}')
