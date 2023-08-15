#!/usr/bin/env python 
# -*- coding:utf-8 -*

import numpy as np
import pandas as pd
import sys
import joblib

input_file = sys.argv[1]
output_file = sys.argv[2]
model_file = 'model.pkl'
target_vol = 100
basic_vol = 2


def get_ms(tm):
    hhmmss = tm // 1000
    ms = (hhmmss // 10000 * 3600 + (hhmmss // 100 % 100) * 60 + hhmmss % 100) * 1000 + tm % 1000
    ms_from_open = ms - 34200000  # millisecond from stock opening
    if tm >= 130000000:
        ms_from_open -= 5400000
    return ms_from_open


tick_data = pd.read_csv(input_file, index_col=None)
symbol = list(tick_data['COLUMN02'].unique())
symbol.sort()
model = joblib.load(model_file)

# simple prediction strategy
od_book = []
for sym in symbol:
    sym_data = tick_data[tick_data['COLUMN02'] == sym]
    cum_vol_buy = 0  # accumulate buying volume
    cum_vol_sell = 0  # accumulate selling volume
    unfinished_buy = 0  # unfinished buying volume in this round
    unfinished_sell = 0  # unfinished selling volume in this round

    index = sym_data['COLUMN01'].values
    ms = sym_data['COLUMN03'].apply(lambda x: get_ms(x)).values
    price = sym_data['COLUMN07'].values
    sampling_p = 0  # sampling pointer
    for i in range(len(ms)):
        if ms[i] < 13800000:  # before 14:50:00
            if ms[i] - ms[sampling_p] < 300000:  # execute the order every 5 minutes
                continue
            sampling_p = i

            # find the indexes at different time
            idx_5min, idx_10min, idx_15min, idx_20min, idx_25min = 0, 0, 0, 0, 0
            for j in range(i):
                if ms[i] - ms[j] >= 300000:
                    idx_5min = j
                if ms[i] - ms[j] >= 600000:
                    idx_10min = j
                if ms[i] - ms[j] >= 900000:
                    idx_15min = j
                if ms[i] - ms[j] >= 1200000:
                    idx_20min = j
                if ms[i] - ms[j] >= 1500000:
                    idx_25min = j

            # calculate the 10 factor variables and make prediction
            if idx_25min != 0:
                x = np.array([
                    price[i] / price[idx_5min] - 1,
                    price[i] / price[idx_10min] - 1,
                    price[i] / price[idx_15min] - 1,
                    price[i] / price[idx_20min] - 1,
                    price[i] / price[idx_25min] - 1,
                    max(price[idx_5min:i]) / min(price[idx_5min:i]) - 1,
                    max(price[idx_10min:i]) / min(price[idx_10min:i]) - 1,
                    max(price[idx_15min:i]) / min(price[idx_15min:i]) - 1,
                    max(price[idx_20min:i]) / min(price[idx_20min:i]) - 1,
                    max(price[idx_25min:i]) / min(price[idx_25min:i]) - 1,
                ]).reshape(1, -1)
                y = model.predict(x)[0]

                if y >= 0:
                    od_vol = basic_vol + unfinished_buy
                    if target_vol - cum_vol_buy >= od_vol:
                        od_book.append([sym, 'B', index[i], od_vol])
                        cum_vol_buy += od_vol
                    else:
                        od_book.append([sym, 'B', index[i], target_vol - cum_vol_buy])
                        cum_vol_buy = target_vol
                    unfinished_buy = 0
                    unfinished_sell += basic_vol
                else:
                    od_vol = basic_vol + unfinished_sell
                    if target_vol - cum_vol_sell >= od_vol:
                        od_book.append([sym, 'S', index[i], od_vol])
                        cum_vol_sell += od_vol
                    else:
                        od_book.append([sym, 'S', index[i], target_vol - cum_vol_sell])
                        cum_vol_sell = target_vol
                    unfinished_sell = 0
                    unfinished_buy += basic_vol
        else:
            if ms[i] - ms[sampling_p] <= 60000:
                continue
            if target_vol - cum_vol_buy > 0:  # force complete before market closes
                od_book.append([sym, 'B', index[i], target_vol - cum_vol_buy])
            if target_vol - cum_vol_sell > 0:  # force complete before market closes
                od_book.append([sym, 'S', index[i], target_vol - cum_vol_sell])
            break

od_book = pd.DataFrame(od_book, columns=['symbol', 'BSflag', 'dataIdx', 'volume'])
od_book.to_csv(output_file, index=False)
