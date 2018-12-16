import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

# reads in the teams data
teams = pd.read_csv(r'..\Baseball Analytics\data\Teams.csv')
teams = teams[teams['yearID'] >= 1985] # only takes teams after 1985
teams = teams[['yearID','teamID','Rank','R','RA','G','W','H','BB','HBP','AB','SF','SB','CS','HR','2B','3B']]
teams = teams.set_index(['yearID','teamID'])

# reads in salaries data
salaries = pd.read_csv(r'..\Baseball Analytics\data\Salaries.csv')
# computes total payroll for every team per year
salariesGrouped = salaries.groupby(['yearID','teamID'])['salary'].sum()
teams = teams.join(salariesGrouped)

# add new data
teams['BA'] = teams['H']/teams['AB']
teams['OBP'] = (teams['H']+teams['BB']+teams['HBP'])/(teams['AB']+teams['BB']+teams['HBP']+teams['SF'])
teams['SLG'] = (teams['H']+teams['2B']+2*teams['3B']+3*teams['HR'])/teams['AB']
# new metrics for slugging, OBP and stolen base percentage
teams['SLG2'] = (0.5*teams['SF']+teams['H']+teams['2B']+2*teams['3B']+3*teams['HR'])/teams['AB']
teams['OBP2'] = (teams['H']+teams['BB']+teams['HBP'])/(teams['AB']+teams['BB']+teams['HBP'])
teams['STL'] = teams['SB']/(teams['SB']+teams['CS'])
# First Model
regModel1 = sm.ols("R~OBP+SLG+BA",teams)
reg1 = regModel1.fit() # this has R^2 = 0.919 and adjusted = 0.918
# Second Model
regModel2 = sm.ols("R~OBP+SLG",teams)
reg2 = regModel2.fit() # this has R^2 = 0.919 and adjusted = 0.919
# Third Model
regModel3 = sm.ols("R~OBP+SLG+STL",teams)
reg3 = regModel3.fit() # this has R^2 = 0.922 and adjusted = 0.921
# Fourth Model
regModel4 = sm.ols("R~OBP2+SLG+STL",teams)
reg4 = regModel4.fit() # this has R^2 = 0.923 and adjusted = 0.922
# Fifth Model
regModel5 = sm.ols("R~OBP2+SLG2+STL",teams)
reg5 = regModel5.fit() # this is best with R^2 = 0.924 and adjusted = 0.924
# Sixth Model
regModel6 = sm.ols("R~OBP2+SLG+STL",teams)
reg6 = regModel6.fit() # this has R^2 = 0.923 and adjusted = 0.922
print(reg1.summary())
print(reg2.summary())
print(reg3.summary())
print(reg4.summary())
print(reg5.summary())
print(reg6.summary())


