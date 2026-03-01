import pandas as pd

# Load the cleaned data
df = pd.read_csv('cleaned_players_list.csv')

# Ensure 'Salary' is an integer
df['Salary'] = df['Salary'].astype(int)

# Remove players with no 'FPPG' value as they cannot contribute to the lineup
df = df.dropna(subset=['FPPG'])

# Optionally, save the DataFrame again after these additional cleaning steps
df.to_csv('preprocessed_players_list.csv', index=False)
