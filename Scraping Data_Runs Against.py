import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

# reads in the teams data
teams = pd.read_csv(r'..\Baseball Analytics\data\Teams.csv')
teams = teams[teams['yearID'] >= 1985] # only takes teams after 1985
teams = teams[['yearID','teamID','Rank','RA','G','ERA','HA','HRA','FP','E']]
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
# new metric for stolen base percentage
teams['STL'] = teams['SB']/(teams['SB']+teams['CS']) 
# First Model
regModel1 = sm.ols("R~OBP+SLG+BA",teams)
reg1 = regModel1.fit() # this has R^2 = 0.919 and adjusted = 0.918
regModel2 = sm.ols("R~OBP+SLG",teams)
reg2 = regModel2.fit() # this has R^2 = 0.919 and adjusted = 0.919
regModel3 = sm.ols("R~OBP+SLG+STL",teams)
reg3 = regModel3.fit() # this is best with R^2 = 0.922, adjusted = 0.921

print(reg1.summary())
print(reg2.summary())
print(reg3.summary())
