import requests
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

with open('config.json') as f:
    config = json.load(f)

# Add your Bing Search V7 subscription key and endpoint to your environment variables.
subscription_key = config['bingAPIKey']
endpoint = "https://api.bing.microsoft.com/v7.0/search"

# Query term(s) to search for.
query = "bitcoin"

# Construct a request
mkt = 'en-US'
headers = {'Ocp-Apim-Subscription-Key': subscription_key}

# Store the results
results = []

# For each of the past 7 days
for i in range(7):
    # Calculate the date
    date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')

    # Add the date to the parameters
    params = {'q': query, 'mkt': mkt, 'count': 50, 'offset': 0, 'freshness': 'Day', 'textFormat': 'Raw', 'setLang': 'EN'}

    # Call the API
    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()

    # Parse the JSON response
    data = response.json()

    # Extract the totalEstimatedMatches value and store it with the date
    results.append((date, data['webPages']['totalEstimatedMatches']))

# Sort the results by date
results.sort()

# Split the results into two lists for plotting
dates, matches = zip(*results)

# Create a plot
plt.plot(dates, matches)
plt.xlabel('Date')
plt.ylabel('Estimated number of matches')
plt.title('Bitcoin usage over time')
plt.show()