import pandas as pd
from pulp import *

# Load the cleaned data
df = pd.read_csv('cleaned_players_list.csv')

# Convert 'Salary' to integers
df['Salary'] = df['Salary'].astype(int)

# Drop players without 'FPPG' as they won't contribute to the lineup
df = df[df['FPPG'].notnull()]

# You may also want to handle other columns that are critical for optimization

# Save the preprocessed data to a new CSV (optional)
df.to_csv('preprocessed_players_list.csv', index=False)

# Print out information to confirm the preprocessing
print(df.info())

# Define the problem
prob = LpProblem("MLBLineupOptimization", LpMaximize)

# Create a list of player IDs to use as decision variables
player_ids = df['Id'].tolist()
player_vars = LpVariable.dicts("Players", player_ids, cat='Binary')  # Binary decision variable

# Objective function: Maximize the sum of projected points
prob += lpSum([df.loc[df['Id'] == pid, 'FPPG'].values[0] * player_vars[pid] for pid in player_ids]), "Total Projected Points"

# Constraint: Stay under the salary cap
salary_cap = 35000  # Example salary cap; change this to your specific cap
prob += lpSum([df.loc[df['Id'] == pid, 'Salary'].values[0] * player_vars[pid] for pid in player_ids]) <= salary_cap, "Salary Cap"

# Add constraints for player positions, team diversity, etc. here...

# Solve the problem
prob.solve()

# Output the selected players
for pid in player_ids:
    if player_vars[pid].value() == 1:
        player = df[df['Id'] == pid]
        print(player[['First Name', 'Last Name', 'Position', 'Salary', 'FPPG']])
