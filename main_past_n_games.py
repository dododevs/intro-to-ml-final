import datetime
import numpy as np
import pandas as pd

# user functions
from load_dataset import load_dataset
from find_related_games import find_any_past_n_games
from learning import random_forest_learning, support_vector_learning, bayes_learning, knn_learning, dummy_learning
from plot import compare_binary_classification

n_games = 50 # number of past games to consider
limit = None # truncate dataframe for testing (saves time) - None for complete df
sample_size = 0   # number of times to train model to get good accuracy mean

df = load_dataset("local/games.csv") # load dataset from csv into dataframe

df_aug = find_any_past_n_games(df, n_games, limit)

# prepare containers for assessment ploting
model_name_plot, prec_rec_plot, pr_auc_plot, fpr_tpr_plot, roc_auc_plot, acc_plot, time_plot  = [[] for _ in range(7)]

## acutall learning
# define params
X_labels = df_aug.keys().values[1:]
y_label = "HOME_TEAM_WINS"
scaler = "MinMax"

# learn random forest
accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption = random_forest_learning(df_aug, X_labels, y_label, scaler, random_state = 26)

# store assessment scores
model_name_plot.append('RF')
acc_plot.append(accuracy)
prec_rec_plot.append([precision, recall])
pr_auc_plot.append(pr_auc)
fpr_tpr_plot.append([fpr, tpr])
roc_auc_plot.append(roc_auc)
time_plot.append(time_consumption)

## learn support vector machine
#accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption = #support_vector_learning(df_aug, X_labels, y_label, scaler, random_state = 26)
#
## store assessment scores
#model_name_plot.append('SVC')
#acc_plot.append(accuracy)
#prec_rec_plot.append([precision, recall])
#pr_auc_plot.append(pr_auc)
#fpr_tpr_plot.append([fpr, tpr])
#roc_auc_plot.append(roc_auc)
#time_plot.append(time_consumption)

# learn naive bayes
accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption = bayes_learning(df_aug, X_labels, y_label, scaler, random_state = 26)

# store assessment scores
model_name_plot.append('NB')
acc_plot.append(accuracy)
prec_rec_plot.append([precision, recall])
pr_auc_plot.append(pr_auc)
fpr_tpr_plot.append([fpr, tpr])
roc_auc_plot.append(roc_auc)
time_plot.append(time_consumption)

# learn k-nearest neighbors
accuracy, precision, recall, pr_auc, fpr, tpr, roc_auc, time_consumption = knn_learning(df_aug, X_labels, y_label, scaler, random_state = 26)

# store assessment scores
model_name_plot.append('KNN')
acc_plot.append(accuracy)
prec_rec_plot.append([precision, recall])
pr_auc_plot.append(pr_auc)
fpr_tpr_plot.append([fpr, tpr])
roc_auc_plot.append(roc_auc)
time_plot.append(time_consumption)

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
