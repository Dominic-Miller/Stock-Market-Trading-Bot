# ML-Trading-Bot
This is a repository consisting of code and documentation for a trading bot using Python that can be used to simulate profits from day trading with a ML algorithm.

The idea behind this project is to develop a web application where users can go in and create their own portfolio of stocks from a handful of available options to 
be traded by a ML bot. The bot will track 1 minute tickers of the stock data and day trade based on the assumption a stock will always fall back to its mean price 
(if a stock deviates too much from its mean buy/sell that stock). Run a simulation on this data and portfolio contents over the previous year and see what the 
earnings would have looked like.

# How to run the Web App
1.Activate the virtual environment
venv\Scripts\Activate.ps1

2.Run the server
 (venv) PS C:<route_to_repo>\ML-Trading-Bot-2> python manage.py runserver
