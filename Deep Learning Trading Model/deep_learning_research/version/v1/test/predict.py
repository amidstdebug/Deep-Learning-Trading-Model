import numpy as np
import pandas as pd
import dask.dataframe as dd
import sys
import math
import random

# import joblib

# input_file = sys.argv[1]
# output_file = sys.argv[2]
# model_file = 'model.pkl'
target_vol = 100
input_file = 'test_20220808/testdata_20220808.csv'
output_file = 'test_20220808/testdata_20220808'


def get_ms(tm):
    hhmmss = tm // 1000
    ms = (hhmmss // 10000 * 3600 + (hhmmss // 100 % 100) * 60 + hhmmss % 100) * 1000 + tm % 1000
    ms_from_open = ms - 34200000  # millisecond from stock opening
    if tm >= 130000000:
        ms_from_open -= 5400000
    return ms_from_open


tick_data = pd.read_csv(input_file, index_col=None)
tick_data_idx = tick_data[["COLUMN01", "COLUMN02"]]
tick_data = tick_data.dropna()
symbol = list(tick_data['COLUMN02'].unique())
symbol.sort()


# model = joblib.load(model_file)
def mean_reversion_strategy(row):
    """
    # Define a function to determine if the stock is overbought or oversold based on its position relative
    to the Bollinger Bands and its RSI value

    :param row:
    :return: 'B' if the stock is overbought, 'S' if the stock is oversold, 'N' if the stock is neither
    """

    if row['bb_bbhi'] == 1 and row['RSI'] > 80:
        return 'S', row['RSI']
    elif row['bb_bbli'] == 1 and row['RSI'] < 20:
        return 'B', row['RSI']
    else:
        return 'N', None


def volume_rng(flag, capt, impt, current_price, max_shares, owed):
    if flag == "S":
        min_sell = 1
        # Need to experiment with this
        # Need to sell enough to cover losses
        # if capt < 0:
        #     min_sell = math.ceil((-capital) / current_price)

        if min_sell > max_shares:
            return max_shares

        if owed <= 10:
            min_sell = owed

        return np.ma.round(np.random.triangular(
          left=min_sell - 0.5,
          mode=min_sell + (impt/20) * (max_shares - min_sell),
          right=max_shares + 0.5,
          size=1)
        ).astype(int)[0]
    else:
        min_buy = 1

        if min_buy > max_shares:
            return max_shares

        if owed <= 10:
            min_buy = owed

        return np.ma.round(np.random.triangular(
            left=min_buy - 0.5,
            mode=min_buy + (impt / 20) * (max_shares - min_buy),
            right=max_shares + 0.5,
            size=1)
        ).astype(int)[0]

trading_capital = 1000000
# risk management

MAX_TOTAL_VOLUME = 100

od_book_risk = []
od_book_volatility = []
od_book_liquidity = []
od_book = []

for sym in symbol:
    # print(sym)
    # if sym != "000009.SZ":
    #     break
    total_volume_risk = 0
    total_volume_volatility = 0
    total_volume_liquidity = 0
    sym_data = tick_data[tick_data['COLUMN02'] == sym]
    stock_ti = {
        "remaining_buys": 100,
        "remaining_sells": 100,
        "owed_buys": 0,
        "owed_sells": 0,
        "last_bought": None,
        "last_sold": None,
        "num_buys": 0,
        "num_sells": 0,
        "current_capital": 0
    }

    sym_idx = tick_data_idx[tick_data_idx['COLUMN02'] == sym]

    for i in range(len(sym_idx)):
        current_idx = sym_idx.iloc[i]["COLUMN01"]

        if sym_data[sym_data["COLUMN01"] == current_idx].shape[0] == 0:
            od_book.append([sym, "N", current_idx, 0])
            continue

        current_row = sym_data[sym_data["COLUMN01"] == current_idx].squeeze()

        # Calculate distance between entry price and stop-loss level using ATR
        stop_loss_distance = current_row['ATR']

        # Set risk tolerance
        risk_tolerance = 0.01

        # Calculate volume based on risk tolerance and stop-loss distance
        volume_risk = (trading_capital * risk_tolerance) / stop_loss_distance

        # Set volatility tolerance
        volatility_tolerance = 0.5

        # Calculate volume based on volatility tolerance and ATR
        volume_volatility = (trading_capital * volatility_tolerance) / current_row['ATR']

        # Set liquidity tolerance
        liquidity_tolerance = 0.1

        # Calculate average liquidity using selling volume columns
        avg_liquidity = current_row[[f'COLUMN{i}' for i in range(18, 28)]].mean()

        # Calculate volume based on liquidity tolerance and average liquidity
        volume_liquidity = avg_liquidity * liquidity_tolerance

        action, cur_rsi = mean_reversion_strategy(current_row)
        if action == "N":
            od_book.append([current_row["COLUMN02"], action, current_row["COLUMN01"], 0])
        else:
            if action == "S" and stock_ti['remaining_sells'] != 0:
                weight = 20 - (100 - cur_rsi)
                vol = volume_rng(
                    action,
                    stock_ti['current_capital'],
                    weight,
                    current_row["COLUMN07"],
                    stock_ti['remaining_sells'],
                    stock_ti['owed_sells']
                )

                stock_ti['remaining_sells'] -= vol
                stock_ti['owed_sells'] -= vol
                stock_ti['owed_buys'] += vol
                stock_ti['current_capital'] += vol * current_row["COLUMN07"]
                od_book.append([current_row["COLUMN02"], "S", current_row["COLUMN01"], vol])
            elif action == "B" and stock_ti['remaining_buys'] != 0:
                weight = 20 - cur_rsi
                vol = volume_rng(
                    action,
                    stock_ti['current_capital'],
                    weight,
                    current_row["COLUMN07"],
                    stock_ti['remaining_buys'],
                    stock_ti['owed_buys'])

                stock_ti['remaining_buys'] -= vol
                stock_ti['owed_buys'] -= vol
                stock_ti['owed_sells'] += vol
                stock_ti['current_capital'] -= vol * current_row["COLUMN07"]
                od_book.append([current_row["COLUMN02"], "B", current_row["COLUMN01"], vol])
            else:
                od_book.append([current_row["COLUMN02"], "N", current_row["COLUMN01"], 0])

        od_book_risk.append([current_row["COLUMN02"], action, current_row["COLUMN01"], volume_risk])
        od_book_volatility.append([current_row["COLUMN02"], action, current_row["COLUMN01"], volume_volatility])
        od_book_liquidity.append([current_row["COLUMN02"], action, current_row["COLUMN01"], volume_liquidity])



od_book_risk_df = pd.DataFrame(od_book_risk, columns=['symbol', 'BSflag', 'dataIdx', 'volume'])
od_book_volatility_df = pd.DataFrame(od_book_volatility, columns=['symbol', 'BSflag', 'dataIdx', 'volume'])
od_book_liquidity_df = pd.DataFrame(od_book_liquidity, columns=['symbol', 'BSflag', 'dataIdx', 'volume'])
od_book_df = pd.DataFrame(od_book, columns=['symbol', 'BSflag', 'dataIdx', 'volume'])

od_book_risk_df.to_csv(output_file + '_risk.csv', index=False)
od_book_volatility_df.to_csv(output_file + '_volatility.csv', index=False)
od_book_liquidity_df.to_csv(output_file + '_liquidity.csv', index=False)

od_book_df.set_index(od_book_df["dataIdx"], inplace=True)
od_book_df = od_book_df.reindex(tick_data_idx["COLUMN01"])
od_book_df.to_csv(output_file + "_ordertime.csv", index=False)
