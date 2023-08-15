import numpy as np
import pandas as pd
import sys
import ta
import time
from collections import deque


class MaxLengthList:
	def __init__(self, max_length):
		self.max_length = max_length
		self._list = []
	
	def append(self, item):
		if len(self._list) == self.max_length:
			self._list.pop(0)
		self._list.append(item)
	
	def __getitem__(self, index):
		return self._list[index]
	
	def __repr__(self):
		return repr(self._list)


def generate_strings():
	result = []
	for i in range(93000000, 113001000, 1000):
		result.append(str(i))
	for i in range(130000000, 150001000, 1000):
		result.append(str(i))
	return result



window_size = 5
tick_info = MaxLengthList(window_size)
tick_data_path = "D:/Personal/Downloads/tickdata_20220805 (1).csv"

with open(tick_data_path, 'r') as f:
	# Skip first line
	f.readline()
	stock_data = {}
	processed_lines = set()
	time_strings = generate_strings()
	start_time_list = time_strings[:window_size]
	line = f.readline().split(',')
	for time in start_time_list:
		while int(line[2]) <= int(time):
			# prevent duplicate rows of data
			if tuple(line[1:6]) not in processed_lines:
				processed_lines.add(tuple(line[1:6]))
				line_stock_code = line[1]
				line_stock_price = line[6]
				if stock_data.get(line_stock_code) is None:
					stock_data[line_stock_code] = deque(['0'] * start_time_list.index(time), maxlen=5)
					stock_data[line_stock_code].append(line_stock_price)
				else:
					stock_data[line_stock_code].append(line_stock_price)
			line = f.readline().split(',')
		# check if stock exists at that particular time index
		for stock_code in stock_data:
			if len(stock_data[stock_code]) < len(start_time_list):
				stock_data[stock_code].append('0')
	print(stock_data)
	
	

# Process initial window
# print(tick_info[0])
# price = tick_info[0][6]
# max_price = tick_info[0][4]
# min_price = tick_info[0][5]
# print(price)
# max_risk = abs((float(price) - float(max_price))/float(max_price)*100)
# min_risk = abs((float(price) - float(min_price))/float(min_price)*100)
# print(max_risk)
# print(min_risk)
# Move window through file
# for line in f:
# 	tick_info.append(line)
# 	# Process current window
# 	print(tick_info)
