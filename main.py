import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
from timeit import default_timer as timer

# user functions
from load_dataset import load_dataset
from find_related_games import find_related_games


n_games_max = 3     # defines how many past games should be considered
limit_df = 800        # truncates the dataset for faster testing

df = load_dataset("local/games.csv") # load dataset from csv into dataframe

# a matrix with each row having the indices of the related games
# first entry in row is the index of the first game, all other entries
# are the indices of games with the same teams playing in the past
R = find_related_games(df, n_games_max, limit_df)
print(R)

# create dataframe in the size of the actual information we can use
indices_of_games_with_enough_data = df.iloc[R[:,0]]
df_useful = df.iloc[R[:,0]].copy()

## Extend dataframe with extra columns for previous game data
# Add new columns to the dataset (for the prevoius game stats)
# for each game, query the game stats and append them to the dataframe as columns
# - loop through each n-game (n-1, n-2, ...), 
# - then loop through each label of the previous game
# - calculate the to-be-added column by iterating through the indices of the n-(i-th) game
# - store that in an array
# - append array as column to dataframe
df_extended = df_useful.copy()
labels = df.keys().values[1:-1] # columns that are to be considered from the previous games
print(f'Considering the following columns from the previous games: {labels}')
for ngame in range(n_games_max):
    for og_label in labels:
        new_column_content = df.iloc[R[:,ngame+1]][og_label].values
        new_column_label = f'{og_label}_n-{ngame+1}'
        
        df_extended[new_column_label] = new_column_content

# debug output:
print(df_extended.head())
print(df_extended.keys())
