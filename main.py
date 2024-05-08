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
from math import sqrt

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
print("2. Compare two cryptocurrencies")
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

        # Get public ip address of this machine
        import requests
        ip = requests.get('https://api64.ipify.org').text
        print(ip)

        try:
            interest_over_time = pytrends.interest_over_time()
            # Save the Google Trends data to a CSV file in the folder "google_trends_data"
            interest_over_time.to_csv(f"google_trends_data/{cryptocurrency}_google_trends.csv")
        except:
            # If there is an error, use previously saved data if available
            try:
                print("Error getting Google Trends data. Using previously saved data...")
                interest_over_time = pd.read_csv(f"google_trends_data/{cryptocurrency}_google_trends.csv")
            except:
                print("Error getting Google Trends data. Please try again later.")
                exit()


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

        # fig.tight_layout()
        plt.title(f"{cryptocurrency} Monthly Price vs Google Trends")
        plt.show()

        # Calculate the correlation between the price and the google trends data
        correlation, p_value = pearsonr(df_monthly["price"], df_monthly["google_trends"])
        print(f"Correlation: {correlation}")  # This number should be between -1 and 1 (0 means no correlation) (Closer to 1 means positive correlation, closer to -1 means negative correlation)
        print(f"P-value: {p_value}")

        # Price prediction based on Google Trends data
        # Split the data into training and testing data
        train_size = int(len(df_monthly) * 0.7)
        train = df_monthly[:train_size]
        test = df_monthly[train_size:]

        # Linear regression model
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from math import sqrt

        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        X = df_monthly[["google_trends"]]
        y = df_monthly["price"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Create the linear regression model
        model = LinearRegression()

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate errors
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)
        rmse = sqrt(mean_squared_error(y_test, predictions))
        print(f"Root Mean Squared Error: {rmse}")
        r2 = r2_score(y_test, predictions)
        print(f"R^2 Score: {r2}")


        # Create a DataFrame from X_test and predictions
        df_predictions = pd.DataFrame(predictions, index=X_test.index, columns=['Predictions'])

        # Sort the DataFrame by date
        df_predictions.sort_index(inplace=True)

        # Plot the sorted predictions
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = 'tab:red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color=color)
        ax1.plot(df_monthly["price"], color=color)
        ax1.plot(df_predictions.index, df_predictions['Predictions'], color="green", linestyle="--")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Google Trends', color=color)
        ax2.plot(df_monthly["google_trends"], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f"{cryptocurrency} Monthly Price vs Google Trends")
        plt.show()


        # Create the linear regression model
        model = LinearRegression()

        # Train the model
        model.fit(train[["google_trends"]], train["price"])

        # Make predictions
        predictions = model.predict(test[["google_trends"]])

        # Calculate the root mean squared error
        rmse = sqrt(mean_squared_error(test["price"], predictions))
        print(f"Root Mean Squared Error: {rmse}")

        # Visualize the predictions with the actual price and the google trends data | Use ax to plot on the same figure
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = 'tab:red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color=color)
        ax1.plot(df_monthly["price"], color=color)
        ax1.plot(test.index, predictions, color="green", linestyle="--")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Google Trends', color=color)
        ax2.plot(df_monthly["google_trends"], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f"{cryptocurrency} Monthly Price vs Google Trends")
        plt.show()

        # Line separation
        print("--------------------------------------------------------")

        # Print the predictions
        print(predictions)

        # Line separation
        print("--------------------------------------------------------")
elif option == "2":
    # Use two cryptocurrencies to compare the relationship between the two and see if it can be predicted
    # if given one price, can we predict the other price? (or near it)
    # Ask the user to choose two cryptocurrencies
    cryptocurrency1 = input("Enter the name of the first cryptocurrency you want to compare: ")
    cryptocurrency2 = input("Enter the name of the second cryptocurrency you want to compare: ")

    # Check if the cryptocurrencies are in the list of cryptocurrencies_data
    if f"{cryptocurrency1}.csv" in cryptocurrencies_data and f"{cryptocurrency2}.csv" in cryptocurrencies_data:
        print(f"Comparing {cryptocurrency1} and {cryptocurrency2}...")

        # Load the data of the chosen cryptocurrencies
        try:
            df1 = pd.read_csv(f"cryptocurrencies_data/{cryptocurrency1}.csv")
        except:
            print(f"Error loading data for {cryptocurrency1}")
            exit()

        try:
            df2 = pd.read_csv(f"cryptocurrencies_data/{cryptocurrency2}.csv")
        except:
            print(f"Error loading data for {cryptocurrency2}")
            exit()


        # Print the first 5 rows of the data
        print(df1.head())
        print(df2.head())

        # Print the last 5 rows of the data
        print(df1.tail())
        print(df2.tail())

        # Clean date stamp to remove unnecessary time
        df1["date"] = df1["date"].apply(lambda x: x.split()[0])
        df2["date"] = df2["date"].apply(lambda x: x.split()[0])

        # Print the first 5 rows of the data
        print(df1.head())
        print(df2.head())

        # Daily change in price
        df1["Daily Change Value"] = df1["price"].diff()
        df2["Daily Change Value"] = df2["price"].diff()

        # Daily percentage change in price
        df1["Price Change %"] = df1["price"].pct_change() * 100
        df2["Price Change %"] = df2["price"].pct_change() * 100

        # Print the first 5 rows of the data
        print(df1.head())
        print(df2.head())

        # Make a line plot for a green line for the positive changes and a red line for the negative changes, do not include the dates because it is too crowded
        plt.figure(figsize=(12, 6))
        plt.plot(df1["Daily Change Value"], color="green", label=f"{cryptocurrency1} Positive Change")
        plt.plot(df1["Daily Change Value"].mask(df1["Daily Change Value"] > 0), color="red",
                 label=f"{cryptocurrency1} Negative Change")
        plt.plot(df2["Daily Change Value"], color="blue", label=f"{cryptocurrency2} Positive Change")
        plt.plot(df2["Daily Change Value"].mask(df2["Daily Change Value"] > 0), color="orange",
                 label=f"{cryptocurrency2} Negative Change ")
        plt.title(f"{cryptocurrency1} and {cryptocurrency2} Daily Change in Price")
        plt.xlabel("Days")
        plt.ylabel("Price Change")
        plt.legend()
        plt.show()

        # Make a line plot for the daily percentage change in price
        plt.figure(figsize=(12, 6))
        plt.plot(df1["Price Change %"], color="red", label=f"{cryptocurrency1}")
        plt.plot(df2["Price Change %"], color="blue", label=f"{cryptocurrency2}")
        plt.title(f"{cryptocurrency1} and {cryptocurrency2} Daily Percentage Change in Price")
        plt.xlabel("Days")
        plt.ylabel("Price Change %")
        plt.legend()
        plt.show()

        # Make a histogram for the daily percentage change in price
        plt.figure(figsize=(12, 6))
        sns.histplot(df1["Price Change %"].dropna(), kde=True, color="red", label=f"{cryptocurrency1}")
        sns.histplot(df2["Price Change %"].dropna(), kde=True, color="blue", label=f"{cryptocurrency2}")
        plt.title(f"{cryptocurrency1} and {cryptocurrency2} Daily Percentage Change in Price")
        plt.xlabel("Price Change %")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

        # Make a scatter plot of the price vs the total_volume
        plt.figure(figsize=(12, 6))
        plt.scatter(df1["price"], df1["total_volume"], color="red", label=f"{cryptocurrency1}")
        plt.scatter(df2["price"], df2["total_volume"], color="blue", label=f"{cryptocurrency2}")
        plt.title(f"{cryptocurrency1} and {cryptocurrency2} Price vs Total Volume")
        plt.xlabel("Price")
        plt.ylabel("Total Volume")
        plt.legend()
        plt.show()

        # Line separation
        print("--------------------------------------------------------")

        # Print the current dataframe of the cryptocurrency data
        print(df1)
        print(df2)

        # Line separation
        print("--------------------------------------------------------")


        df1["date"] = pd.to_datetime(df1["date"])
        df1.set_index("date", inplace=True)
        df2["date"] = pd.to_datetime(df2["date"])
        df2.set_index("date", inplace=True)
        # List of columns to resample
        cols_to_resample = ['price', 'total_volume', 'Daily Change Value', 'Price Change %']

        # Resample the data into monthly data and calculate the mean for each month
        df1_monthly = df1[cols_to_resample].resample('MS').mean()
        df2_monthly = df2[cols_to_resample].resample('MS').mean()

        # Comapre the oldest dates and the newest dates to get the range of dates to remove the rows that are not in both dataframes
        start_date = max(df1_monthly.index.min(), df2_monthly.index.min())
        end_date = min(df1_monthly.index.max(), df2_monthly.index.max())

        # Convert Timestamp to string in the required format
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        print("Start Date:", start_date)
        print("End Date:", end_date)

        print("New Dataframes:")
        # Modify the dataframes to only include the dates that are in both dataframes
        df1_monthly = df1_monthly[(df1_monthly.index >= start_date) & (df1_monthly.index <= end_date)]
        df2_monthly = df2_monthly[(df2_monthly.index >= start_date) & (df2_monthly.index <= end_date)]
        print(df1_monthly)
        print(df2_monthly)

        # Using the features of the first cryptocurrency to predict the price of the second cryptocurrency
        # Then using the features of the second cryptocurrency to predict the price of the first cryptocurrency
        # Create the linear regression model

        # imports
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from math import sqrt

        # First cryptocurrency to predict the second cryptocurrency
        X = df1_monthly[["price", "total_volume", "Daily Change Value", "Price Change %"]]
        y = df2_monthly["price"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        # Create the linear regression model
        model = LinearRegression()

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate errors
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)
        rmse = sqrt(mean_squared_error(y_test, predictions))
        print(f"Root Mean Squared Error: {rmse}")
        r2 = r2_score(y_test, predictions)
        print(f"R^2 Score: {r2}")

        # Convert predictions to a DataFrame
        predictions_df = pd.DataFrame(predictions, index=X_test.index, columns=['Predictions'])

        # Sort X_test and predictions_df by the index
        X_test_sorted = X_test.sort_index()
        predictions_df_sorted = predictions_df.sort_index()

        # Plot the sorted data
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = 'tab:red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel(f"{cryptocurrency2} Price", color=color)
        ax1.plot(df2_monthly["price"], color=color)
        ax1.plot(X_test_sorted.index, predictions_df_sorted['Predictions'], color="green", linestyle="--")
        ax1.tick_params(axis='y', labelcolor=color)

        # Legend
        ax1.legend(["Actual Price", "Predicted Price"])

        plt.title(f"{cryptocurrency1} Features to Predict {cryptocurrency2} Price")
        plt.show()

        # Second cryptocurrency to predict the first cryptocurrency
        X = df2_monthly[["price", "total_volume", "Daily Change Value", "Price Change %"]]
        y = df1_monthly["price"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        # Create the linear regression model
        model2 = LinearRegression()

        # Train the model
        model2.fit(X_train, y_train)

        # Make predictions
        predictions = model2.predict(X_test)

        # Calculate errors
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)
        rmse = sqrt(mean_squared_error(y_test, predictions))
        print(f"Root Mean Squared Error: {rmse}")
        r2 = r2_score(y_test, predictions)
        print(f"R^2 Score: {r2}")

        # Convert predictions to a DataFrame
        predictions_df = pd.DataFrame(predictions, index=X_test.index, columns=['Predictions'])

        # Sort X_test and predictions_df by the index
        X_test_sorted = X_test.sort_index()
        predictions_df_sorted = predictions_df.sort_index()

        # Plot the sorted data
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = 'tab:red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel(f"{cryptocurrency1} Price", color=color)
        ax1.plot(df1_monthly["price"], color=color)
        ax1.plot(X_test_sorted.index, predictions_df_sorted['Predictions'], color="green", linestyle="--")
        ax1.tick_params(axis='y', labelcolor=color)

        # Legend
        ax1.legend(["Actual Price", "Predicted Price"])

        plt.title(f"{cryptocurrency2} Features to Predict {cryptocurrency1} Price")
        plt.show()

        # Make visualizations to show the accuracy of the predictions using a error plot
        # First cryptocurrency to predict the second cryptocurrency
        # Calculate the errors
        errors = y_test - predictions
        # Sort errors by the index (date)
        errors_sorted = errors.sort_index()

        # Plot the sorted errors
        plt.figure(figsize=(12, 6))
        plt.plot(errors_sorted, color="red")
        plt.title(f"{cryptocurrency1} Features to Predict {cryptocurrency2} Price Errors")
        plt.xlabel("Date")
        plt.ylabel("Error")
        plt.show()

        # print predictions with the date index
        print(predictions_df_sorted)



elif option == "3":
    print("Exiting...")


else:
    print("Cryptocurrency not found. Please enter a valid cryptocurrency.")
