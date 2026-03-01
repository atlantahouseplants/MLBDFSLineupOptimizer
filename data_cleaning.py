import pandas as pd

# Load the dataset
df = pd.read_csv('FanDuel-MLB-2024 ET-03 ET-28 ET-100355-players-list.csv')

# Print the first few rows of the DataFrame to understand what the data looks like
print(df.head())

# Print a summary of the DataFrame to identify any missing values and the data types of each column
print(df.info())

# Replace NaN values with 0 (or another appropriate value) in the columns that are numeric
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = df[numeric_columns].fillna(0)

# Convert columns to the correct data types if necessary (e.g., salary might need to be a float)
# df['Salary'] = df['Salary'].astype(float)

# If there are categorical columns with NaN that you need to fill with a placeholder:
# df['Position'] = df['Position'].fillna('Unknown')

# Remove rows with missing values if they cannot be replaced with a default value
# df = df.dropna()

# Save the cleaned data to a new CSV file
cleaned_csv_path = 'cleaned_players_list.csv'
df.to_csv(cleaned_csv_path, index=False)
print(f'Cleaned data saved to {cleaned_csv_path}')
