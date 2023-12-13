import pandas as pd

# Read dataset from csv and returns a sorted dataframe (by time)
def load_dataset(path_to_dataset):
    df = pd.read_csv(path_to_dataset, parse_dates=["GAME_DATE_EST"], dtype={
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

# number the rows consecutively from 0 on, due to the ordering according to date, this
# assures, that when searching in higher indexed rows, search is always done in past games
    df.reset_index(inplace=True)
    print("Original data info:")
    df.info()
    print("\n")
    return df