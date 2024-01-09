# Final project for Intro to ML course

Dataset is at [Kaggle](https://www.kaggle.com/datasets/nathanlauga/nba-games)

Interesting literature and websites:

 - [Predicting the Outcome of NBA Games Matthew Houde - Bryant Graduate](https://digitalcommons.bryant.edu/cgi/viewcontent.cgi?article=1000&context=honors_data_science)
 - [Which NBA Statistics Actually Translate to Wins?](https://watchstadium.com/which-nba-statistics-actually-translate-to-wins-07-13-2019/)
 - [Building My first Machine Learning Model | NBA Prediction Algorithm](https://towardsdatascience.com/building-my-first-machine-learning-model-nba-prediction-algorithm-dee5c5bc4cc1)
 - [Bunch of NBA related GitHub Repos](https://github.com/topics/nba-prediction)
 - [Predict NBA games, make money - machine learning project](https://towardsdatascience.com/predict-nba-games-make-money-machine-learning-project-b222b33f70a3)
 - [Predicting NBA Game Outcomes](https://cs229.stanford.edu/proj2017/final-reports/5231214.pdf)
 - [Predicting the Outcome of NBA Games Using Machine Learing](https://medium.com/nerd-for-tech/predicting-the-outcome-of-nba-games-using-machine-learning-676a62549040)

# Quick code overview
## main_test_player_stats.py
Just for testing the approach with including the player stats, can be ignored

## NBA_DataExploration.ipynb
From the beginning of the project, can be ignored

## main_past_n_games
Code by Eugenio and Andrea and Theo, just sorts the last games of the team against any opponents

## main_same_oppenents.py
Last n_games of a team against the opposing team

### learning.py
used by main_past_n_games and main_same_opponents
scaling, dividing train, test set
NO CV! TODO

### load_dataset
Loads and sorts the game data by date

### plot.py
Plots the results and some performance indices