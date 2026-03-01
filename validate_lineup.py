# Replace the following variables with the actual values
SALARY_CAP = 35000
REQUIRED_PLAYERS = {'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3}
TOTAL_PLAYERS = 9

# Load your optimized lineup - ensure it matches the format of your actual data
optimized_lineup_df = pd.read_csv('optimized_lineup.csv')

# Validation
total_salary = optimized_lineup_df['Salary'].sum()
total_players = optimized_lineup_df.shape[0]
players_by_position = optimized_lineup_df['Position'].value_counts()

# Check salary cap
if total_salary <= SALARY_CAP:
    print(f"Valid lineup: Total salary ${total_salary} is within the cap of ${SALARY_CAP}.")
else:
    print(f"Invalid lineup: Total salary ${total_salary} exceeds the cap of ${SALARY_CAP}.")

# Check player positions
for position, count in REQUIRED_PLAYERS.items():
    if players_by_position.get(position, 0) == count:
        print(f"Valid lineup: Correct number of {position} players.")
    else:
        print(f"Invalid lineup: Incorrect number of {position} players.")

# Check total players
if total_players == TOTAL_PLAYERS:
    print(f"Valid lineup: Correct total number of players ({TOTAL_PLAYERS}).")
else:
    print(f"Invalid lineup: Incorrect total number of players ({total_players} instead of {TOTAL_PLAYERS}).")
