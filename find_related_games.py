import numpy as np
import os

def find_related_games(df, n_games_max, limit=None, force_recalculation=False):
    """Takes a dataframe and returns a matrix with the indices of the games and the indices of their previous games"""
    # column 1 contains the indices of the games
    # remaining columns contain the indices of the previous game(s)

    filepath=f'/local/temp' # relative path to saved R matrix
    filename=f'prev_games_matrix_limit_{str(limit)}'

    # check if related games have been calculated already
    try:
        R = np.load(f'{os.getcwd()}{filepath}/{filename}.npy')
        print(f'Found previously calculated related games matrix with limit {limit} at {os.getcwd()}{filepath}/{filename}')
        return R

    except:
        print(f'Did not find a previously calculated related games matrix with limit {limit} at {filepath}')
        # reduce size for testing on smaller dataset
        if limit==None:
            df_qpg = df.copy()
        if df.index.size > limit:
            df_qpg = df.iloc[:limit].copy()
        else:
            raise Exception('Set limit exceeds dataframe size.')

        # initialize return matrix
        R = np.empty((0,n_games_max+1),int)
        # Rt = np.vstack((Wt,[[1, 2, 3, 4]]))

        # iterate through all games
        for index, current_row in df_qpg[['HOME_TEAM_ID','VISITOR_TEAM_ID']].iterrows():
            #print(f'index: {index}\n{current_row}\n size of query-field: {df_qpg[["HOME_TEAM_ID","VISITOR_TEAM_ID"]].iloc[index+1:].size}\n\n')
            
            prev_game_counter = 0   # to count how many previous matches were found
            same_game_indices = np.array((index),int)    # initialize array of indices with current game
            
            # query for previous games
            for index_query, past_row in df_qpg[['HOME_TEAM_ID','VISITOR_TEAM_ID']].iloc[index+1:].iterrows():
            # print(f'query index: {index_query}\n')
                current_home_id = current_row['HOME_TEAM_ID']
                previous_home_id = past_row['HOME_TEAM_ID']
                current_away_id = current_row['VISITOR_TEAM_ID']
                previous_away_id = past_row['VISITOR_TEAM_ID']

                if (current_home_id==previous_home_id and 
                current_away_id==previous_away_id):
                    prev_game_counter+=1
                    same_game_indices = np.append(same_game_indices, [index_query])
                    # print(f'{prev_game_counter}. PREVIOUS MATCH of game {index} FOUND at index {index_query}')
                if prev_game_counter >= n_games_max:
                    R = np.vstack((R,same_game_indices))
                    print(f'FOUND all the desired {n_games_max} previous games for game {index}')
                    print(f'same game indices: {same_game_indices}')
                    print(f'{df.iloc[same_game_indices]}') # show ORIGINAL df rows to compare results
                    break
        try:
            os.mkdir(f'{os.getcwd()}{filepath}')
        except:
            print(f'{os.getcwd()}{filepath} already exists')
        finally:
            np.save(f'{os.getcwd()}{filepath}/{filename}', R)
        return R