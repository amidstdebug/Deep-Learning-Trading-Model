# Context

We have to make a trading bot to maximise a daily return. 

## Round 1
### Running Phase
From March 6 to March 24, they will generate a random tickdata file 
calculate the score, and then show the ranking.

These scores are for us to essentially judge how good our model is,
and the date generated for the tickdata file is random. It may be
inside the 100-day training data or outside, which is similar to 
the final round.

### Closed-Door Phase
From March 27 to March 31, they will test our model on REAL LIFE data 
and average our return over the 5 days.

## Final Round
## Pre-Ranking Phase
From April 10 to April 24, basically same thing happens as the running phase.

## Closed-Door Phase
From April 24 - April 28, same thing happens as the closed-door phase.


# Data

We have 100 days worth of data available to train the model.

We have 500 types of stock to trade.

There must be 100 trades of EACH stock per day, and the total number of trades
is 200 (100 Buy + 100 Sell).

There must be minimum 3 transactions per stock per day, etc:

10 + 10 + 80

OR

30 + 30 + 40

OR

20 + 20 + 20 + 20 + 20

OR

10 + 10 + 10 + 10 + 10 + 10 + 10 + 10 + 10 + 10

In essence, there will be:
100 of each stock * 500 Stocks * 2 (Buy/Sell) = 100,000 transactions per day.

Therefore:
100,000 * 100 days = 10,000,000 transactions in total.

Note: Our trading window is open from 9:30 to 11:30 AM and 1:30 to 3:00 PM.

Note: When using test.py to trade, we can only use data from current time and prior, not future data. 
e.g. if we are trading at 10:00 AM, we can only use data from 9:30 to 10:00 AM.