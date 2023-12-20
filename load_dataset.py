import pandas as pd

# Read dataset from csv and returns a sorted dataframe (by time)
# added ascending param because shifting in past n games requires ascending, not descending date order
def load_dataset(path_to_dataset, ascending = True):
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

    df = df.sort_values("GAME_DATE_EST", ascending = ascending)

    print("Original data info:")
    df.info()
    print("\n")
    return df





# Read dataset with individual stats per player from csv and returns a sorted dataframe (by 1. game 2. team 3. player)
def load_players_dataset(path_to_dataset):
    df = pd.read_csv(path_to_dataset,
                     dtype={
                        "GAME_ID": int,
                        "TEAM_ID": int,
                        "PLAYER_ID": int,
                     })

    # Drop redundant columns and entries with null columns, and cols without "meaning"
    df = df.drop(["TEAM_ABBREVIATION",
                  "TEAM_CITY",
                  "PLAYER_NAME",
                  "NICKNAME",
                  "START_POSITION",
                  "COMMENT",
                  "MIN"], axis=1)
    df = df.dropna() # drops a about 20% of the rows but those contain no usefull stats

    # Sort by descending GAME_ID, TEAM_ID, PLAYER_ID
    df = df.sort_values(["GAME_ID", "TEAM_ID", "PLAYER_ID"], ascending = False)

    # number the rows consecutively from 0 on
    df.reset_index(inplace=True)
    print("Original data info:")
    df.info()
    print("\n")
    return df
