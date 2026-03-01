import pandas as pd

# Replace 'your_csv_file.csv' with the path to the CSV file you downloaded
df = pd.read_csv('FanDuel-MLB-2024 ET-03 ET-28 ET-100355-players-list.csv')

# Check the first few rows of the DataFrame to confirm it's loaded correctly
print(df.head())
