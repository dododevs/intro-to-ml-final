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


indices_of_games_with_enough_data = df.iloc[R[:,0]]
df_useful = df.iloc[R[:,0]].copy()







"""

## Extend dataframe with extra columns for previous game data
# Add new columns to the dataset (for the prevoius game stats)
df_extended = df.copy()
labels = df.keys().values[1:-1] # columns that are to be considered from the previous games
print(f'Considering the following columns from the previous games: {labels}')
for ngame in range(n_games_max):
    for column in labels:
        new_column = f'{column}_n-{ngame+1}'
        if new_column not in df:
            df_extended[new_column] = None # Initialize column if it does not exist
        
        # iterate through the games and fill cell with data
        for index in range(len(R[:,1])):
            print('hi')
            index_of_current_game = R[index,0]
            index_of_previous_game = R[index,ngame]
            df_extended.iloc[index_of_current_game][new_column]=df.iloc[index_of_previous_game][column]
            print('hi')

df_extended.head()
df_filled = df_extended.copy() # dataframe to be filled with entries of previous games

print(df_extended.head())



#for index in indices_of_games_with_enough_data:



"""


### TODO:
# select only rows with from the dataset that are relevant
# fill up dataframe with previous game values
#
# or
# 
# for each game, query the game stats and append them to the dataframe as columns
#
# or
#
# write a function that gets the whole row of data for a game, including the previous

