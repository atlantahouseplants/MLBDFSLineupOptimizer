from pydfs_lineup_optimizer import get_optimizer, Site, Sport

# Initialize the optimizer for FanDuel MLB
optimizer = get_optimizer(Site.FANDUEL, Sport.BASEBALL)

# Load players from your CSV file
optimizer.load_players_from_csv(r"C:\Users\wallg\MLBDFSLineupOptimizer\cleaned_players_list.csv")

# Generate 10 optimized lineups
lineup_generator = optimizer.optimize(n=10)

# Print each lineup
for lineup in lineup_generator:
    print(lineup)
