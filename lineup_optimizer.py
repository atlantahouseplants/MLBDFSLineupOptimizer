from southpaw import FanduelClient

# Initialize the client
client = FanduelClient()

# Example: Get upcoming contests for MLB
contests = client.upcoming_contests(sport='MLB')
print(contests)
