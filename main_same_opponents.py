import datetime
import numpy as np
import pandas as pd

# user functions
from load_dataset import load_dataset
from find_related_games import find_related_games
from learning import random_forest_learning, support_vector_learning, bayes_learning, knn_learning, dummy_learning
from plot import compare_binary_classification

import sys


n_games_max = 10     # defines how many past games should be considered
limit_df = None     # truncates the dataset for faster testing, set to None if no limit is wanted
sample_size = 0   # number of times to train model to get good accuracy mean

df = load_dataset("local/games.csv", ascending = False) # load dataset from csv into dataframe
# removed dropping of GAME_DATE_EST in load_dataset so it's done here
df.drop("GAME_DATE_EST", axis = 1, inplace = True)

# number the rows consecutively from 0 on, due to the ordering according to date, this
# assures, that when searching in higher indexed rows, search is always done in past games
df.reset_index(inplace=True)

# a matrix with each row having the indices of the related games
# first entry in row is the index of the first game, all other entries
# are the indices of games with the same teams playing in the past
R = find_related_games(df, n_games_max, limit_df)
# print(R)

## Extend dataframe with extra columns for previous game data
# Add new columns to the dataset (for the prevoius game stats)
# for each game, query the game stats and append them to the dataframe as columns
# - loop through each n-game (n-1, n-2, ...), 
# - then loop through each label of the previous game
# - calculate the to-be-added column by iterating through the indices of the n-(i-th) game
# - store that in an array
# - append array as column to dataframe
df_aug = df.iloc[R[:,0]].copy()
labels = df.keys().values[1:-1] # columns that are to be considered from the previous games
print(f'Considering the following columns from the previous games: {labels[4:]}')

# drop current game information (except of HOME_TEAM_WINS)
df_aug = df_aug.drop(labels, axis=1)

for ngame in range(n_games_max):
    for og_label in labels[4:]:
        new_column_content = df.iloc[R[:,ngame+1]][og_label].values
        new_column_label = f'{og_label}_n-{ngame+1}'
        
        df_aug[new_column_label] = new_column_content

df_aug.drop("index", axis = 1, inplace = True)

# debug output:
print(df_aug.head())
print(df_aug.keys())

# calc mean of previous stats
df_aug["prev_PTS_home"] = df_aug[[s for s in df_aug.columns if 'PTS_home_' in s]].sum(axis=1) / n_games_max
df_aug["prev_FG_PCT_home"] = df_aug[[s for s in df_aug.columns if 'FG_PCT_home_' in s]].sum(axis=1) / n_games_max
df_aug["prev_FT_PCT_home"] = df_aug[[s for s in df_aug.columns if 'FT_PCT_home_' in s]].sum(axis=1) / n_games_max
df_aug["prev_FG3_PCT_home"] = df_aug[[s for s in df_aug.columns if 'FG3_PCT_home_' in s]].sum(axis=1) / n_games_max
df_aug["prev_AST_home"] = df_aug[[s for s in df_aug.columns if 'AST_home_' in s]].sum(axis=1) / n_games_max
df_aug["prev_REB_home"] = df_aug[[s for s in df_aug.columns if 'REB_home_' in s]].sum(axis=1) / n_games_max

df_aug["prev_PTS_away"] = df_aug[[s for s in df_aug.columns if 'PTS_away_' in s]].sum(axis=1) / n_games_max
df_aug["prev_FG_PCT_away"] = df_aug[[s for s in df_aug.columns if 'FG_PCT_away_' in s]].sum(axis=1) / n_games_max
df_aug["prev_FT_PCT_away"] = df_aug[[s for s in df_aug.columns if 'FT_PCT_away_' in s]].sum(axis=1) / n_games_max
df_aug["prev_FG3_PCT_away"] = df_aug[[s for s in df_aug.columns if 'FG3_PCT_away_' in s]].sum(axis=1) / n_games_max
df_aug["prev_AST_away"] = df_aug[[s for s in df_aug.columns if 'AST_away_' in s]].sum(axis=1) / n_games_max
df_aug["prev_REB_away"] = df_aug[[s for s in df_aug.columns if 'REB_away_' in s]].sum(axis=1) / n_games_max

#drop previous stats
end_index = n_games_max * 12
df_aug.drop(df_aug.keys().values[1:end_index+1], axis = 1, inplace = True)

# debug output:
print(df_aug.head())
print(df_aug.keys())
df_aug.info()

# prepare containers for assessment ploting
model_name_plot, prec_rec_plot, pr_auc_plot, fpr_tpr_plot, roc_auc_plot, acc_plot, time_plot  = [[] for _ in range(7)]

### Actual machine learning
# define params
X_labels = df_aug.keys().values[1:]
y_label = "HOME_TEAM_WINS"
scaler = "MinMax"

min_samples_split = [2] # This parameter does not seem to make a difference
max_depth = list(range(6,9,1))         # change max_depth parameter here
print(f'Iterating through RF parameters:\n  min_samples_split:{min_samples_split}\n  max_depth:{max_depth}')

for mins in min_samples_split:
    for maxd in max_depth:
        print(f'min_samples_split: {mins} max_depth: {maxd}')

        # learn random forest
        accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption = random_forest_learning(df_aug, X_labels, y_label, scaler, random_state = 26, min_samples_split=mins, max_depth=maxd)

        # store assessment scores
        model_name_plot.append(f'RF{maxd}')
        acc_plot.append(accuracy)
        prec_rec_plot.append([precision, recall])
        pr_auc_plot.append(pr_auc)
        fpr_tpr_plot.append([fpr, tpr])
        roc_auc_plot.append(roc_auc)
        time_plot.append(time_consumption)

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
# accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption = # bayes_learning(df_aug, X_labels, y_label, scaler, random_state = 26)
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
# accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption = # knn_learning(df_aug, X_labels, y_label, scaler, random_state = 26)
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
# learn dummy
accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption = dummy_learning(df_aug, X_labels, y_label, scaler, random_state = 26)

# store rf assessment scores
model_name_plot.append('DUMMY')
acc_plot.append(accuracy)
prec_rec_plot.append([precision, recall])
pr_auc_plot.append(pr_auc)
fpr_tpr_plot.append([fpr, tpr])
roc_auc_plot.append(roc_auc)
time_plot.append(time_consumption)

compare_binary_classification(model_name_plot, prec_rec_plot, pr_auc_plot, fpr_tpr_plot, roc_auc_plot, acc_plot, time_plot)

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
