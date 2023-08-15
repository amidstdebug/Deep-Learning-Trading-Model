The current strategy makes use of the Mean Reversion Strategy.

A mean reversion strategy is a type of trading strategy that relies on the idea that prices tend to revert back to their long-term average or mean over time. The strategy involves identifying when a stock, commodity, or other asset is trading significantly above or below its historical average, and then taking a position in the expectation that the price will eventually move back towards the mean.

The Mean Reversion Strategy makes use of the RSI and Bollinger Band technical indicators to signal when to buy or sell.

The script holds off from selling or buy for the last and first 15 minutes.

The volume bought/sold is determined by a triangle distribution, with the mode being determined by the RSI.

Work flow:
	1. Run the comp.py file with the tickdata as the first argument, and the output file as the second argument
