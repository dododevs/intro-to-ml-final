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

file_path = os.path.abspath(os.path.join('output','past_n_games.csv'))
with open(file_path, 'a') as out_file:
    n_games = int(sys.argv[1]) # number of past games to consider - used command line args to write bash script iteration...
    limit = None # truncate dataframe for testing (saves time) - None for complete df
    sample_size = 0   # number of times to train model to get good accuracy mean

    df = load_dataset("local/games.csv") # load dataset from csv into dataframe

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

    # define a list of all features (with the home/away) always paired, without pts and fg_pts as they always seem important (saves time)
    # then iterate through every possible subset of the list
    # and delete that subset of features. Part of the grid search for optimal params
    # and done in absence of thorough statistical influence analysis
    all_features = [["prev_FT_PCT_home", "prev_FT_PCT_away"], ["prev_FG3_PCT_home", "prev_FG3_PCT_away"],
                    ["prev_AST_home", "prev_AST_away"], ["prev_FT_PCT_diff"], ["prev_FG3_PCT_diff"], ["prev_AST_diff"]]

    # all_features = [[]] # use this to delete no feature

    power_set = chain.from_iterable(combinations(all_features, r) for r in range(len(all_features) + 1))

    for delete_features in power_set:
        # force file writing to avoid loss of data if something breaks
        out_file.flush()
        os.fsync(out_file)

        delete_features_flattened = [item for l in delete_features for item in l]

        df_work = df_aug.drop(delete_features_flattened, axis = 1)

        # grr = pd.plotting.scatter_matrix(df_work, c=df_work["HOME_TEAM_WINS"]) # can plot the scatter matrix if wanted

        # # prepare containers for assessment ploting
        # model_name_plot, prec_rec_plot, pr_auc_plot, fpr_tpr_plot, roc_auc_plot, acc_plot, time_plot  = [[] for _ in range(7)]

        ## acutall learning
        # define params
        X_labels = df_work.keys().values[1:]
        y_label = "HOME_TEAM_WINS"
        scaler = sys.argv[2] # scaling used in preprocessing MinMax or Standard - use command line args to write bash script iteration...

        # learn random forest
        accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption, params = random_forest_learning(df_work, X_labels, y_label, scaler=scaler, random_state = 26)

        # write results into csv to access them later
        out_string = f'{n_games}; rf; {df_work.keys().values}; {scaler}; {accuracy}; {pr_auc}; {roc_auc}; {time_consumption}; {params}'
        out_string = out_string.replace("\n", "")
        out_file.write(out_string + "\n")

        # # store assessment scores
        # model_name_plot.append(f'RF')
        # acc_plot.append(accuracy)
        # prec_rec_plot.append([precision, recall])
        # pr_auc_plot.append(pr_auc)
        # fpr_tpr_plot.append([fpr, tpr])
        # roc_auc_plot.append(roc_auc)
        # time_plot.append(time_consumption)

        # learn support vector machine
        accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption, params = support_vector_learning(df_work, X_labels, y_label, scaler, random_state = 26)

        # write results into csv to access them later
        out_string = f'{n_games}; svc; {df_work.keys().values}; {scaler}; {accuracy}; {pr_auc}; {roc_auc}; {time_consumption}; {params}'
        out_string = out_string.replace("\n", "")
        out_file.write(out_string + "\n")

        # # store assessment scores
        # model_name_plot.append('SVC')
        # acc_plot.append(accuracy)
        # prec_rec_plot.append([precision, recall])
        # pr_auc_plot.append(pr_auc)
        # fpr_tpr_plot.append([fpr, tpr])
        # roc_auc_plot.append(roc_auc)
        # time_plot.append(time_consumption)

        # learn naive bayes
        accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption, params = bayes_learning(df_work, X_labels, y_label, scaler, random_state = 26)

        # write results into csv to access them later
        out_string = f'{n_games}; nb; {df_work.keys().values}; {scaler}; {accuracy}; {pr_auc}; {roc_auc}; {time_consumption}; {params}'
        out_string = out_string.replace("\n", "")
        out_file.write(out_string + "\n")

        # # store assessment scores
        # model_name_plot.append('NB')
        # acc_plot.append(accuracy)
        # prec_rec_plot.append([precision, recall])
        # pr_auc_plot.append(pr_auc)
        # fpr_tpr_plot.append([fpr, tpr])
        # roc_auc_plot.append(roc_auc)
        # time_plot.append(time_consumption)

        # learn k-nearest neighbors
        accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption, params = knn_learning(df_work, X_labels, y_label, scaler, random_state = 26)

        # write results into csv to access them later
        out_string = f'{n_games}; knn; {df_work.keys().values}; {scaler}; {accuracy}; {pr_auc}; {roc_auc}; {time_consumption}; {params}'
        out_string = out_string.replace("\n", "")
        out_file.write(out_string + "\n")

        # # store assessment scores
        # model_name_plot.append('KNN')
        # acc_plot.append(accuracy)
        # prec_rec_plot.append([precision, recall])
        # pr_auc_plot.append(pr_auc)
        # fpr_tpr_plot.append([fpr, tpr])
        # roc_auc_plot.append(roc_auc)
        # time_plot.append(time_consumption)

        # learn dummy
        accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption, params = dummy_learning(df_work, X_labels, y_label, scaler, random_state = 26)

        # write results into csv to access them later
        out_string = f'{n_games}; dummy; {df_work.keys().values}; {scaler}; {accuracy}; {pr_auc}; {roc_auc}; {time_consumption}; {params}'
        out_string = out_string.replace("\n", "")
        out_file.write(out_string + "\n")

        # # store assessment scores
        # model_name_plot.append('DUMMY')
        # acc_plot.append(accuracy)
        # prec_rec_plot.append([precision, recall])
        # pr_auc_plot.append(pr_auc)
        # fpr_tpr_plot.append([fpr, tpr])
        # roc_auc_plot.append(roc_auc)
        # time_plot.append(time_consumption)

        # compare_binary_classification(model_name_plot, prec_rec_plot, pr_auc_plot, fpr_tpr_plot, roc_auc_plot, acc_plot, time_plot)

        # learn model multiple times for statistical analysis of results
        if sample_size > 0:
            store = np.empty([len(max_depth), sample_size])
            for i in range(len(max_depth)):
                for j in range(sample_size):
                    print(f'Learning model {j+1}/{sample_size} with max_depth = {max_depth[i]}/{max_depth}')
                    accuracy, _, _, _, _, _, _, _ = random_forest_learning(df_work, X_labels, y_label, scaler, random_state = 26, max_depth=max_depth[i])
                    store[i][j] = accuracy

            max_acci = [0, 0] # first entry is index and second is maximum of accuracies
            for i in range(len(max_depth)):
                acc_mean = np.mean(store[i])
                if acc_mean > max_acci[0]:
                    max_acci = [i, acc_mean]
            print(f'Maximum accuracy is {max_acci[1]} with max_depth = {max_depth[max_acci[0]]}')
