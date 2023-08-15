import pandas as pd
import ta
from datetime import datetime
import time
import os

# Get the current path
current_path = os.getcwd()

# Define a new path to join
directory = "tickdata"

# Join the new path with the current path
directory = os.path.join(current_path, directory)

# Define the path to the directory
filename_list = []
# Loop through all files in the directory
for filename in os.listdir(directory):
	# Check if the file is a CSV file
	if filename.endswith(".csv"):
		# If it is, print the filename
		filename_list.append(filename)

filename_list.sort()


def train_engineer_features(filename):
	"""
	Not to be run in real time!
	"""
	data = pd.read_csv(filename)
	# sort by column 2 then 3 (symbol then time)
	data = data.sort_values(by=['COLUMN02', 'COLUMN03'])
	
	# convert the 'COLUMN03' column to string to remove useless zeros
	data['COLUMN03'] = data['COLUMN03'].astype(str).str[:-3]
	
	# convert the 'COLUMN03' column back to int
	data['COLUMN03'] = data['COLUMN03'].astype(int)
	
	# # make a list to divide columns 04 to 17, 28-37, 52-55 by 10000 to get correct price values
	# columns = ['COLUMN04', 'COLUMN05', 'COLUMN06', 'COLUMN07', 'COLUMN08', 'COLUMN09', 'COLUMN10', 'COLUMN11',
	#            'COLUMN12', 'COLUMN13', 'COLUMN14', 'COLUMN15', 'COLUMN16', 'COLUMN17', 'COLUMN28', 'COLUMN29',
	#            'COLUMN30', 'COLUMN31', 'COLUMN32', 'COLUMN33', 'COLUMN34', 'COLUMN35', 'COLUMN36', 'COLUMN37',
	#            'COLUMN52', 'COLUMN53', 'COLUMN54', 'COLUMN55']
	#
	# # divide the columns by 10000
	# data[columns] = data[columns] / 10000
	
	# # # convert the 'COLUMN03' column to datetime format
	# data['COLUMN03'] = pd.to_datetime(data['COLUMN03'], format='%H%M%S')
	
	final_data = []
	
	# add features per symbol
	for symbol in data['COLUMN02'].unique():
		symbol_data = data[data['COLUMN02'] == symbol].copy()
		
		# drop first row
		symbol_data = symbol_data.drop(symbol_data.index[0])
		
		# begin to engineer features
		
		symbol_data['diff'] = symbol_data['COLUMN07'].diff()
		symbol_data['gain'] = symbol_data['diff'].clip(lower=0).round(2)
		symbol_data['loss'] = symbol_data['diff'].clip(upper=0).round(2)
		
		# instantiate Bollinger Bands indicator
		ta_bbands = ta.volatility.BollingerBands(close=symbol_data["COLUMN07"],
		                                         window=20,
		                                         window_dev=2)
		
		# add Bollinger Bands
		symbol_data['bb_bbm'] = ta_bbands.bollinger_mavg()
		symbol_data['bb_bbh'] = ta_bbands.bollinger_hband()
		symbol_data['bb_bbl'] = ta_bbands.bollinger_lband()
		
		symbol_data['bb_bbhi'] = ta_bbands.bollinger_hband_indicator()
		symbol_data['bb_bbli'] = ta_bbands.bollinger_lband_indicator()
		symbol_data['bb_bbw'] = ta_bbands.bollinger_wband()
		symbol_data['bb_bbp'] = ta_bbands.bollinger_pband()
		
		symbol_data["bb_ma"] = ta_bbands.bollinger_mavg()
		symbol_data["bb_high"] = ta_bbands.bollinger_hband()
		symbol_data["bb_low"] = ta_bbands.bollinger_lband()
		
		# instantiate RSI indicator
		ta_RSI = ta.momentum.RSIIndicator(close=symbol_data["COLUMN07"], window=14)
		symbol_data['RSI'] = ta_RSI.rsi()
		
		# instantiate MACD indicator
		ta_macd = ta.trend.MACD(close=symbol_data["COLUMN07"], window_slow=26, window_fast=12, window_sign=9)
		symbol_data['MACD'] = ta_macd.macd()
		symbol_data['MACD_signal'] = ta_macd.macd_signal()
		symbol_data['MACD_diff'] = ta_macd.macd_diff()
		
		# instantiate Stochastic Oscillator indicator
		ta_stoch = ta.momentum.StochasticOscillator(high=symbol_data["COLUMN05"], low=symbol_data["COLUMN06"],
		                                            close=symbol_data["COLUMN07"],
		                                            window=14, smooth_window=3)
		symbol_data['stoch_k'] = ta_stoch.stoch()
		symbol_data['stoch_d'] = ta_stoch.stoch_signal()
		
		# instantiate ATR indicator
		ta_atr = ta.volatility.AverageTrueRange(high=symbol_data["COLUMN05"], low=symbol_data["COLUMN06"],
		                                        close=symbol_data["COLUMN07"],
		                                        window=14)
		symbol_data['ATR'] = ta_atr.average_true_range()
		
		# instantiate RMA indicator
		ta_rma = ta.trend.WMAIndicator(close=symbol_data["COLUMN07"], window=30)
		symbol_data['RMA'] = ta_rma.wma()
		
		final_data.append(symbol_data)
	final_data = pd.concat(final_data)
	final_data = final_data.sort_values(by=['COLUMN03', 'COLUMN02'])
	return final_data


total_time = 0

for filename in filename_list:
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	print("Current Time =", current_time)
	print('enter:', filename)
	
	start_time = time.time()  # start time for current file
	
	new_csv = train_engineer_features('tickdata/' + filename)
	new_filename = filename.replace("tickdata", "testdata")
	new_csv.to_csv(os.path.join(current_path, 'engineered') + '/' + new_filename, index=False)
	
	end_time = time.time()  # end time for current file
	file_time = end_time - start_time  # time taken for current file
	total_time += file_time  # add current file time to total time
	
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	print("Current Time =", current_time)
	print('exit:', filename)
	print('time taken for', filename, ':', file_time, 'seconds')  # print time taken for current file

print('total time taken:', total_time, 'seconds')


def realtime_engineer_features(filename):
	"""
	to be run in realtime!
	"""
# need to finish training then we add to this function
# need to rethink how we calculate the features and make the trades in real time
# might need to make 3 bots for:
# 1. price prediction
# 2. trade execution timing
# 3. trade execution volume

# be sure to drop first row though
