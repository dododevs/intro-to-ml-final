import datetime
import pandas as pd
from dataclasses import dataclass
from timeit import default_timer as timer


# all statistics for one team in one single game wrapped in a nice data type
@dataclass
class GameStats:
    date: datetime
    game_id: int
    team_id: int
    pts: float
    fg_pct: float
    ft_pct: float
    fg3_pct: float
    ast: float
    reb: float

    def to_list(self) -> list:
        return [self.pts, self.fg_pct, self.ft_pct, self.fg3_pct, self.ast, self.reb]



df = pd.read_csv("local/games.csv", parse_dates=["GAME_DATE_EST"], dtype={
    "HOME_TEAM_ID": int,
    "VISITOR_TEAM_ID": int,
    "GAME_ID": int,
    "SEASON": int,
    "HOME_TEAM_WINS": int
})

# Drop redundant columns and entries with null columns
df = df.drop(["GAME_STATUS_TEXT", "TEAM_ID_home", "TEAM_ID_away"], axis=1)
df = df.dropna()

# Sort by descending date
df = df.sort_values("GAME_DATE_EST", ascending = False)

# compactify dataset (duplicates game date and game id for later comparison)
df["HOME_STATS"] = [GameStats(row[0], row[1], row[2], row[3], row[4], row[5],
                              row[6], row[7], row[8])
                    for row
                    in zip(df["GAME_DATE_EST"],
                           df["GAME_ID"],
                           df["HOME_TEAM_ID"],
                           df["PTS_home"],
                           df["FG_PCT_home"],
                           df["FT_PCT_home"],
                           df["FG3_PCT_home"],
                           df["AST_home"],
                           df["REB_home"])]

df = df.drop(["PTS_home",
              "FG_PCT_home",
              "FT_PCT_home",
              "FG3_PCT_home",
              "AST_home",
              "REB_home"], axis = 1)

df["AWAY_STATS"] = [GameStats(row[0], row[1], row[2], row[3], row[4], row[5],
                              row[6], row[7], row[8])
                    for row
                    in zip(df["GAME_DATE_EST"],
                           df["GAME_ID"],
                           df["VISITOR_TEAM_ID"],
                           df["PTS_away"],
                           df["FG_PCT_away"],
                           df["FT_PCT_away"],
                           df["FG3_PCT_away"],
                           df["AST_away"],
                           df["REB_away"])]

df = df.drop(["PTS_away",
              "FG_PCT_away",
              "FT_PCT_away",
              "FG3_PCT_away",
              "AST_away",
              "REB_away"], axis = 1)

# number the rows consecutively from 0 on, due to the ordering according to date, this
# assures, that when searching in higher indexed rows, search is always done in past games
df.reset_index(inplace=True)
print("Original data info:")
df.info()
print("\n")

# head just for testing on smaller dataset -> faster, should be removed if used as final code
df_aug = df.copy()

start = timer()
tuples = []

# iterate through all rows, for each row search in past games for the first row, where
# home and visitor have the same id as current rows home and visitor respectively
# safe the current and past id as a tuple in a big list (a dictionary like datastructure
# should be more efficient, adapt if possible)
for i, row_current in df_aug.iterrows():
    for _, row_past in df_aug.iloc[i+1:].iterrows():
        if (row_current.iloc[3] == row_past.iloc[3]) and (row_current.iloc[4] == row_past.iloc[4]):
            tuples.append((row_current.iloc[2], row_past.iloc[2]))
            break
end = timer()

print('time for nested for: ' +  str(end - start) + "\n")
print(str(len(tuples)) + "\n")

# set index as GAME_ID, so you can extract all relevant information
# from the dataframe by the game ids stored in the previous step
# renames GAME_ID to index and drops the 0.. index col
df_aug.set_index("GAME_ID", inplace=True)
# df_aug.info()

# first attempt to set up the final dataframe, where a row holds the
# result of the current game, as well as the stats aof the previous one
tmp = []

col_names = [# "HOME_TEAM_ID",
             # "VISITOR_TEAM_ID",
             # "GAME_DATE",
             # "SEASON",
             "HOME_TEAM_WINS",
             "HOME_STATS",
             "AWAY_STATS"]

for (current_id, past_id) in tuples:
    # print(current_id)
    current_row = df_aug.loc[[current_id]] #current_row = df_aug.iloc[current_id]
    past_row = df_aug.loc[[past_id]] #past_row = df_aug.iloc[past_id]
    tmp.append([# current_row['HOME_TEAM_ID'].values[0], # HOME_TEAM_ID
                # current_row['VISITOR_TEAM_ID'].values[0], # VISITOR_TEAM_ID
                # current_row['GAME_DATE_EST'].values[0],
                # current_row['SEASON'].values[0], # SEASON
                current_row['HOME_TEAM_WINS'].values[0], # HOME_TEAM_WINS
                past_row['HOME_STATS'].values[0], # HOME_STATS
                past_row['AWAY_STATS'].values[0]] # AWAY_STATS
    )

df_tmp = pd.DataFrame(tmp, columns = col_names)

df_tmp = df_tmp.dropna()

# list comprehension - gets all previous matches in ok time but needs
# post processing which could also be expensiv
# start = timer()
# tuples = [(row_current[2], row_past[2])
#           for index, row_current in df_aug.iterrows()
#           for _, row_past in df_aug.iloc[index+1:].iterrows()
#           if ((row_current[3] == row_past[3]) and (row_current[4] == row_past[4]))]
# #  or ((row_current[3] == row_past[4]) and (row_current[4] == row_past[3]))
# end = timer()
# print('time for list comprehension: ' + str(end - start))
# print(len(tuples))

# unpack past game stats
df_tmp[["prev_PTS_home",
        "prev_FG_PCT_home",
        "prev_FT_PCT_home",
        "prev_FG3_PCT_home",
        "prev_AST_home",
        "prev_REB_home"]] = [stats.to_list() for stats in df_tmp["HOME_STATS"]]

df_tmp[["prev_PTS_away",
        "prev_FG_PCT_away",
        "prev_FT_PCT_away",
        "prev_FG3_PCT_away",
        "prev_AST_away",
        "prev_REB_away"]] = [stats.to_list() for stats in df_tmp["AWAY_STATS"]]

df_tmp.drop(["HOME_STATS", "AWAY_STATS"], axis = 1, inplace = True)
print("final data:")
df_tmp.info()

print("Saving to file!!!\n")
df_tmp.to_csv('local/games_with_past_stats.csv')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df_work = df_tmp.copy()

y = df_work["HOME_TEAM_WINS"]
X = df_work.drop("HOME_TEAM_WINS", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_rf_class = rf_classifier.predict(X_test)
rf_class_accuracy = accuracy_score(y_rf_class, y_test)

print(rf_class_accuracy)
