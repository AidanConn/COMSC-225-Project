####################################################################################################
# Name: Aidan Connaughton
# Date: 4/13/2024
# Assignment: Final Project
# Course: COMSC.225.01-24/SP Introduction to Data Science
# Description: Dynamic cryptocurrency analysis tool that uses Google Trends data to predict the price of a cryptocurrency of your choice.
####################################################################################################
# Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr  # If you need to calculate correlation coefficients
from pytrends.request import TrendReq  # If you want to access Google Trends data
from datetime import datetime

# ** Step 1: Load the data* *
# Find all cryptocurrencies_data in the folder "cryptocurrencies_data" and store them in a list
cryptocurrencies_data = []
for file in os.listdir("cryptocurrencies_data"):
    if file.endswith(".csv"):
        cryptocurrencies_data.append(file)

# Print the list of cryptocurrencies_data
print("Available cryptocurrencies_data:")
print(cryptocurrencies_data)

# Print menu of options (Choose a cryptocurrency, all cryptocurrencies, or exit)
print("Menu:")
print("1. Analyze a cryptocurrency")
print("2. Analyze all cryptocurrencies")
print("3. Exit")

# Ask the user to choose an option
option = input("Enter the number of the option you want to choose: ")



if option == "1":
    # Ask the user to choose a cryptocurrency
    cryptocurrency = input("Enter the name of the cryptocurrency you want to analyze: ")

    # Check if the cryptocurrency is in the list of cryptocurrencies_data
    if f"{cryptocurrency}.csv" in cryptocurrencies_data:
        print(f"Analyzing {cryptocurrency}...")

        # Load the data of the chosen cryptocurrency
        df = pd.read_csv(f"cryptocurrencies_data/{cryptocurrency}.csv")

        # Print the first 5 rows of the data
        print(df.head())

        # Print the last 5 rows of the data
        print(df.tail())

        # Print the shape of the data 
    else:
        print("Cryptocurrency not found. Please enter a valid cryptocurrency.")
