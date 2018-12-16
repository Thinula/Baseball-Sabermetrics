import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import numpy as np

players = pd.read_csv(r'..\Baseball Analytics\data\Batting.csv')
players = players[(players['yearID'] >= 1985) & (players['AB'] > 0)]
players = players[['yearID','teamID','playerID','H','AB','BB','HBP','SF','H','2B','3B','HR','SB','CS']]
players.set_index(['playerID','yearID','teamID'])
# for some reason players['H'] was duplicated..
players['Hit'] = players['H'].T.drop_duplicates().T
# add new data
players['BA'] = players['Hit']/players['AB']
players['OBP'] = (players['Hit']+players['BB']+players['HBP'])/(players['AB']+players['BB']+players['HBP'])
players['SLG'] = (0.5*players['SF']+players['Hit']+players['2B']+2*players['3B']+3*players['HR'])/players['AB']
players['STL'] = players['SB']/(players['SB']+players['CS'])

playerSalaries = pd.read_csv(r'..\Baseball Analytics\data\Salaries.csv')
playerSalaries = playerSalaries[playerSalaries['yearID'] >= 1985]
playerSalaries.set_index(['yearID','teamID','playerID'])
salaries = pd.read_csv(r'..\Baseball Analytics\data\Salaries.csv')
# computes total payroll for every team per year
salariesGrouped = salaries.groupby(['yearID','teamID'])['salary'].sum() 
#playerSalaries.join(salariesGrouped)

# this creates new variable for percentage of total team payroll each person got
for play, team, year in players.iterrows():
    print(play,team,year)
