import time

import requests
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

with open('config.json') as f:
    config = json.load(f)

# Add your Bing Search V7 subscription key and endpoint to your environment variables.
subscription_key = config['bingAPIKey']
endpoint = "https://api.bing.microsoft.com/v7.0/search"

# Bing News Search API
subscription_key_news = config['bingAPIKey']
endpoint_news = "https://api.bing.microsoft.com/v7.0/news/search"

# Search term the term bitcoin
search_term = "bitcoin"

