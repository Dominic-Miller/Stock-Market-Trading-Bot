# ML-Trading-Bot

## Overview
This is a repository consisting of code and documentation for a stock market trading bot using Python that can 
be used to simulate profits from day trading with a ML algorithm. When run over one-minute ticker data for both
classes of Google's stock in 2022, this bot returned over 25% in profits for the year as shown in the graph below.

The "Stock Market Trading Bot" is an application designed to empower users with the ability to create 
and manage personalized stock portfolios. Utilizing machine learning algorithms, the bot continually
monitors one-minute stock price data for stocks with multiple classes. It then executes day trading 
strategiesbased on the assumption that stock prices between these two classes will regress to their mean 
values when deviating significantly. Additionally, the system conducts simulations over stock data from 
the year 2022 to provide users with insights into potential earnings. 

## Implementation
To implement this trading bot, I have developed a SVM Model (Support Vector Machine). SVM is a popular and 
effective machine learning algorithm for both classification and regression. The idea of SVM is to construct 
a hyperplane to seperate the two classes of data with the gap being as wide as possible. 

## Motivation
Beyond portfolio management and algorithmic trading, the web application delivers real-time market data and 
stock-related news. The project's motivation stems from the desire to simulate real-time financial markets, 
offering users valuable insights into stock changes and visual representations of market fluctuations. The 
automated trading capabilities save time and enable the implementation of intricate trading strategies. 
Operating around the clock, the bot executes trades when users are unavailable and offers the ability to 
backtest trading strategies with historical data while mitigating irrational trading decisions influenced by 
emotions. It will provide a clear and intuitive dashboard that shows users the overview of the portfolio's 
performance, recent trades, and market trends presenting data in an easily digestable form.

## Graph
Graph of profit/loss percentage versus time for Google over 2022 using this bot:
<img width="601" alt="Google Returns 2022" src="https://github.com/Dominic-Miller/Stock-Market-Trading-Bot/assets/98434708/faf16cfe-9e07-4cd0-87fe-547c8ae926e0">
