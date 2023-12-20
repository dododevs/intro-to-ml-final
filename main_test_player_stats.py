import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
from timeit import default_timer as timer
from statistics import fmean
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split

# user functions
from load_dataset import load_players_dataset
#from find_related_games import find_related_games
#from learning import random_forest_learning


#n_games_max = 2     # defines how many past games should be considered
#limit_df = 4000        # truncates the dataset for faster testing, set to None if no limit is wanted

df = load_players_dataset("local/games_details.csv") # load dataset from csv into dataframe

# extract a set (no duplicates) of all player IDs
all_player_ids = df["PLAYER_ID"].unique()

# calculate average stats for all players
player_stats = {"PLAYER_ID": all_player_ids,
                "FGM":        [fmean(df.loc[df["PLAYER_ID"] == id]["FGM"])        for id in all_player_ids],
                "FGA":        [fmean(df.loc[df["PLAYER_ID"] == id]["FGA"])        for id in all_player_ids],
                "FG_PCT":     [fmean(df.loc[df["PLAYER_ID"] == id]["FG_PCT"])     for id in all_player_ids],
                "FG3M":       [fmean(df.loc[df["PLAYER_ID"] == id]["FG3M"])       for id in all_player_ids],
                "FG3A":       [fmean(df.loc[df["PLAYER_ID"] == id]["FG3A"])       for id in all_player_ids],
                "FG3_PCT":    [fmean(df.loc[df["PLAYER_ID"] == id]["FG3_PCT"])    for id in all_player_ids],
                "FTM":        [fmean(df.loc[df["PLAYER_ID"] == id]["FTM"])        for id in all_player_ids],
                "FTA":        [fmean(df.loc[df["PLAYER_ID"] == id]["FTA"])        for id in all_player_ids],
                "FT_PCT":     [fmean(df.loc[df["PLAYER_ID"] == id]["FT_PCT"])     for id in all_player_ids],
                "OREB":       [fmean(df.loc[df["PLAYER_ID"] == id]["OREB"])       for id in all_player_ids],
                "DREB":       [fmean(df.loc[df["PLAYER_ID"] == id]["DREB"])       for id in all_player_ids],
                "REB":        [fmean(df.loc[df["PLAYER_ID"] == id]["REB"])        for id in all_player_ids],
                "AST":        [fmean(df.loc[df["PLAYER_ID"] == id]["AST"])        for id in all_player_ids],
                "STL":        [fmean(df.loc[df["PLAYER_ID"] == id]["STL"])        for id in all_player_ids],
                "BLK":        [fmean(df.loc[df["PLAYER_ID"] == id]["BLK"])        for id in all_player_ids],
                "TO":         [fmean(df.loc[df["PLAYER_ID"] == id]["TO"])         for id in all_player_ids],
                "PF":         [fmean(df.loc[df["PLAYER_ID"] == id]["PF"])         for id in all_player_ids],
                "PTS":        [fmean(df.loc[df["PLAYER_ID"] == id]["PTS"])        for id in all_player_ids],
                "PLUS_MINUS": [fmean(df.loc[df["PLAYER_ID"] == id]["PLUS_MINUS"]) for id in all_player_ids],
               }

df_player_stats = pd.DataFrame(player_stats)

df_player_stats.info()

sample_df_train, sample_df_test = train_test_split(sample_df, train_size=0.6)

cluster = KMeans(n_clusters=8,
                 n_init=10,
                 precompute_distances='auto',
                 n_jobs=1)
cluster.fit(sample_df_train)
result = cluster.predict(sample_df_test)
