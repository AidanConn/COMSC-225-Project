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

        # Clean date stamp to remove unnecessary time
        df["date"] = df["date"].apply(lambda x: x.split()[0])

        # Print the first 5 rows of the data
        print(df.head())

        # Daily change in price
        df["Daily Change Value"] = df["price"].diff()

        # Daily percentage change in price
        df["Price Change %"] = df["price"].pct_change() * 100

        # Print the first 5 rows of the data
        print(df.head())

        # Make a line plot for a green line for the positive changes and a red line for the negative changes, do not include the dates because it is too crowded
        plt.figure(figsize=(12, 6))
        plt.plot(df["Daily Change Value"], color="green", label="Positive Change")
        plt.plot(df["Daily Change Value"].mask(df["Daily Change Value"] > 0), color="red", label="Negative Change")
        plt.title(f"{cryptocurrency} Daily Change in Price")
        plt.xlabel("Days")
        plt.ylabel("Price Change")
        plt.legend()
        plt.show()

        # Make a line plot for the daily percentage change in price
        plt.figure(figsize=(12, 6))
        plt.plot(df["Price Change %"])
        plt.title(f"{cryptocurrency} Daily Percentage Change in Price")
        plt.xlabel("Days")
        plt.ylabel("Price Change %")
        plt.show()

        # Make a histogram for the daily percentage change in price
        plt.figure(figsize=(12, 6))
        sns.histplot(df["Price Change %"].dropna(), kde=True)
        plt.title(f"{cryptocurrency} Daily Percentage Change in Price")
        plt.xlabel("Price Change %")
        plt.ylabel("Frequency")
        plt.show()

        # Make a scatter plot of the price vs the total_volume
        plt.figure(figsize=(12, 6))
        plt.scatter(df["price"], df["total_volume"], color="blue")
        plt.title(f"{cryptocurrency} Price vs Total Volume")
        plt.xlabel("Price")
        plt.ylabel("Total Volume")
        plt.show()

        # # Now get the Google Trends data how far the data goes back (Test with basic example)
        # pytrends = TrendReq(hl="en-US", tz=360)
        # kw_list = [cryptocurrency]
        # pytrends.build_payload(kw_list, cat=0, timeframe="all", geo="", gprop="")
        # interest_over_time = pytrends.interest_over_time()
        # print(interest_over_time)

        # Do it again but this limit the data to last dates of the cryptocurrency data (Date format in data is "2015-01-04")
        pytrends = TrendReq(hl="en-US", tz=360)
        kw_list = [cryptocurrency]
        start_date = df["date"].min()
        end_date = df["date"].max()
        pytrends.build_payload(kw_list, cat=0, timeframe=f"{start_date} {end_date}", geo="", gprop="")
        interest_over_time = pytrends.interest_over_time()
        print(interest_over_time)

        # Make a line plot of the Google Trends data
        plt.figure(figsize=(12, 6))
        plt.plot(interest_over_time[cryptocurrency], color="purple")
        plt.title(f"{cryptocurrency} Google Trends Data")
        plt.xlabel("Date")
        plt.ylabel("Interest")
        plt.show()

        # Line separation
        print("--------------------------------------------------------")

        # Print the current dataframe of the cryptocurrency data
        print(df)

        # Print the current dataframe of the Google Trends data
        print(interest_over_time)

        # Line separation
        print("--------------------------------------------------------")

        # Make the cryptocurrency data from daily to monthly like the Google Trends data
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        # List of columns to resample
        cols_to_resample = ['price', 'total_volume', 'Daily Change Value',
                            'Price Change %']  # Add other numeric columns if needed

        # Resample the data into monthly data and calculate the mean for each month
        df_monthly = df[cols_to_resample].resample('MS').mean()
        print(df_monthly)

        # Line separation
        print("--------------------------------------------------------")

        # Add the Google Trends data to the monthly cryptocurrency in a new column called "google_trends"
        df_monthly["google_trends"] = interest_over_time[cryptocurrency].resample("MS").mean()
        print(df_monthly)

        # Visualize the monthly data
        # Make a line plot of the monthly price
        plt.figure(figsize=(12, 6))
        plt.plot(df_monthly["price"], color="blue")
        plt.title(f"{cryptocurrency} Monthly Price")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.show()

        # Vidulaize the interest compared to the price
        # Make a line plot of the monthly price and the google trends data
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = 'tab:red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color=color)
        ax1.plot(df_monthly["price"], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Google Trends', color=color)
        ax2.plot(df_monthly["google_trends"], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        #fig.tight_layout()
        plt.title(f"{cryptocurrency} Monthly Price vs Google Trends")
        plt.show()

        # Calculate the correlation between the price and the google trends data
        correlation, p_value = pearsonr(df_monthly["price"], df_monthly["google_trends"])
        print(f"Correlation: {correlation}") # This number should be between -1 and 1 (0 means no correlation) (Closer to 1 means positive correlation, closer to -1 means negative correlation)
        print(f"P-value: {p_value}")

















    else:
        print("Cryptocurrency not found. Please enter a valid cryptocurrency.")
