import pandas as pd
from timeit import default_timer as timer
import matplotlib.pyplot as plt

df = pd.read_csv("local/games_with_past_stats.csv", dtype={"HOME_TEAM_WINS": int})

Y = df["HOME_TEAM_WINS"]
X = df.drop("HOME_TEAM_WINS", axis=1)

grr = pd.plotting.scatter_matrix(df, c=Y, figsize=(50, 50), marker='o',
                                 hist_kwds={'bins': 20}, s=60, alpha=.8)
plt.savefig('scatter.png')
