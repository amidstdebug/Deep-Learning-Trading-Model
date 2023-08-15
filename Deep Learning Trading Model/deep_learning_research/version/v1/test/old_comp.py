import numpy as np
import pandas as pd
import sys
import ta
import time
import math
from datetime import datetime

tick_data_path = sys.argv[1]
order_time_path = sys.argv[2]
start_time = time.time()


# --- Functions ---
def exp_func(x):
    return (math.exp(0.2 * x) - 1) / (math.exp(0.2*20) - 1)
def get_ms(tm):
    hhmmss = tm // 1000
    ms = (hhmmss // 10000 * 3600 + (hhmmss // 100 % 100) * 60 + hhmmss % 100) * 1000 + tm % 1000
    ms_from_open = ms - 34200000  # millisecond from stock opening
    if tm >= 130000000:
        ms_from_open -= 5400000
    return ms_from_open


def mean_reversion_strategy(row):
    """
    # Define a function to determine if the stock is overbought or oversold based on its position relative
    to the Bollinger Bands and its RSI value

    :param row:
    :return: 'B' if the stock is overbought,'S' if the stock is oversold,'N' if the stock is neither
    """

    if row['bb_bbhi'] == 1 and row['rsi'] > 80 and row['cci'] > 100:
        return 'S'
    elif row['bb_bbli'] == 1 and row['rsi'] < 20 and row['cci'] < -100:
        return 'B'
    else:
        return 'N'


def random_volume(flag, impt, sym):
    """
    # Randomly generate a volume to buy/sell
    :param impt: The importance of the transaction based on the RSI
    :param flag: 'B' if buying,'S' if selling
    :param stock: The stock symbol
    :return: volume
    """

    if flag == "S":
        deduction = 0
        if stock_transaction_info[sym]['sell_transactions'] == 0:
            deduction = 3
        elif stock_transaction_info[sym]['sell_transactions'] == 1:
            deduction = 2
        max_sells = stock_transaction_info[sym]['remaining_sells'] - deduction
        return np.ma.round(np.random.triangular(
            left=1 - 0.5,
            mode=1 + (impt / 20) * (max_sells - 1),
            right=max_sells + 0.5,
            size=1)
        ).astype(int)[0]
    elif flag == "B":
        deduction = 0
        if stock_transaction_info[sym]['buy_transactions'] == 0:
            deduction = 3
        elif stock_transaction_info[sym]['buy_transactions'] == 1:
            deduction = 2
        max_buys = stock_transaction_info[sym]['remaining_buys'] - deduction
        return np.ma.round(np.random.triangular(
            left=1 - 0.5,
            mode=1 + (impt / 20) * (max_buys - 1),
            right=max_buys + 0.5,
            size=1)
        ).astype(int)[0]


# --- Main ---
tick_data = open(tick_data_path, 'r')
tick_data.readline()
order_time = open(order_time_path, 'w')
order_time.writelines('symbol,BSflag,dataIdx,volume\n')
order_time.flush()
stock_transaction_info = {}
stock_ta = {}

# --- Loop through each line in tick_data ---
while True:
    # --- Perform feature engineering per stock ---
    current_line = tick_data.readline()

    # Break condition
    if current_line.strip() == '' or len(current_line) == 0:
        break

    current_row = current_line.split(',')
    current_row_ta = {
        "idx": current_row[0],
        "stock": current_row[1],
        "time": int(current_row[2]),
        "price": float(current_row[6]),
        "high": float(current_row[4]),
        "low": float(current_row[5]),
        "rsi": None,
        "bb_bbhi": None,
        "bb_bbli": None
    }
    current_ms_time = get_ms(current_row_ta["time"])
    print(
        f'Index: \x1b[34m{current_row_ta["idx"]}\x1b[0m \x1b[0m(\x1b[32m{(current_ms_time / 14400000) * 100:.2f}\x1b[0m%), Time: \x1b[33m{(time.time() - start_time):.2f}s\x1b[0m',
        end='\r')
    # Initialize stock_transaction_info and stock_ta
    if current_row_ta["stock"] not in stock_transaction_info:
        stock_transaction_info[current_row_ta["stock"]] = {
            "remaining_buys": 100,
            "remaining_sells": 100,
            "owed_buys": 0,
            "owed_sells": 0,
            "time_last_bought": None,
            "time_last_sold": None,
            "buy_transactions": 0,
            "sell_transactions": 0,
            "current_earnings": 0
        }

    if current_row_ta["stock"] not in stock_ta:
        stock_ta[current_row_ta["stock"]] = {
            "window": [],
            "high_window": [],
            "low_window": []
        }

    # Pass if remaining buys and sells are 0
    if stock_transaction_info[current_row_ta["stock"]]["remaining_buys"] == 0 and \
            stock_transaction_info[current_row_ta["stock"]]["remaining_sells"] == 0:
        order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
        order_time.flush()
        continue

    # Finish owed transactions if final few ticks (4 hours from opening)
    if current_ms_time >= 14100000 and (stock_transaction_info[current_row_ta["stock"]]["buy_transactions"] == 0 or
                                        stock_transaction_info[current_row_ta["stock"]]["sell_transactions"] == 0):
        # print("\n", current_row_ta["time"], "\n")
        if stock_transaction_info[current_row_ta["stock"]]["owed_buys"] > 0 and \
                stock_transaction_info[current_row_ta["stock"]]["buy_transactions"] == 0:
            order_time.writelines(
                f'{current_row_ta["stock"]},B,{current_row_ta["idx"]},{int(stock_transaction_info[current_row_ta["stock"]]["owed_buys"] / 3)}\n')
            order_time.flush()
            stock_transaction_info[current_row_ta["stock"]]["remaining_buys"] -= int(
                stock_transaction_info[current_row_ta["stock"]]["owed_buys"] / 3)
            stock_transaction_info[current_row_ta["stock"]]["owed_buys"] -= int(
                stock_transaction_info[current_row_ta["stock"]]["owed_buys"] / 3)
            stock_transaction_info[current_row_ta["stock"]]["buy_transactions"] += 1
            continue
        elif stock_transaction_info[current_row_ta["stock"]]["owed_sells"] > 0 and \
                stock_transaction_info[current_row_ta["stock"]]["sell_transactions"] == 0:
            order_time.writelines(
                f'{current_row_ta["stock"]},S,{current_row_ta["idx"]},{int(stock_transaction_info[current_row_ta["stock"]]["owed_sells"] / 3)}\n')
            order_time.flush()
            stock_transaction_info[current_row_ta["stock"]]["remaining_sells"] -= int(
                stock_transaction_info[current_row_ta["stock"]]["owed_sells"] / 3)
            stock_transaction_info[current_row_ta["stock"]]["owed_sells"] -= int(
                stock_transaction_info[current_row_ta["stock"]]["owed_sells"] / 3)
            stock_transaction_info[current_row_ta["stock"]]["sell_transactions"] += 1
            continue

    if current_ms_time >= 14250000 and (stock_transaction_info[current_row_ta["stock"]]["buy_transactions"] == 1 or
                                        stock_transaction_info[current_row_ta["stock"]]["sell_transactions"] == 1):
        if stock_transaction_info[current_row_ta["stock"]]["owed_buys"] > 0 and \
                stock_transaction_info[current_row_ta["stock"]]["buy_transactions"] == 1:
            order_time.writelines(
                f'{current_row_ta["stock"]},B,{current_row_ta["idx"]},{int(stock_transaction_info[current_row_ta["stock"]]["owed_buys"] / 2)}\n')
            order_time.flush()
            stock_transaction_info[current_row_ta["stock"]]["remaining_buys"] -= int(
                stock_transaction_info[current_row_ta["stock"]]["owed_buys"] / 2)
            stock_transaction_info[current_row_ta["stock"]]["owed_buys"] -= int(
                stock_transaction_info[current_row_ta["stock"]]["owed_buys"] / 2)
            stock_transaction_info[current_row_ta["stock"]]["buy_transactions"] += 1
            continue
        elif stock_transaction_info[current_row_ta["stock"]]["owed_sells"] > 0 and \
                stock_transaction_info[current_row_ta["stock"]]["sell_transactions"] == 1:
            order_time.writelines(
                f'{current_row_ta["stock"]},S,{current_row_ta["idx"]},{int(stock_transaction_info[current_row_ta["stock"]]["owed_sells"] / 2)}\n')
            order_time.flush()
            stock_transaction_info[current_row_ta["stock"]]["remaining_sells"] -= int(
                stock_transaction_info[current_row_ta["stock"]]["owed_sells"] / 2)
            stock_transaction_info[current_row_ta["stock"]]["owed_sells"] -= int(
                stock_transaction_info[current_row_ta["stock"]]["owed_sells"] / 2)
            stock_transaction_info[current_row_ta["stock"]]["sell_transactions"] += 1
            continue

    if current_ms_time >= 14350000:
        if stock_transaction_info[current_row_ta["stock"]]["owed_buys"] > 0:
            order_time.writelines(
                f'{current_row_ta["stock"]},B,{current_row_ta["idx"]},{stock_transaction_info[current_row_ta["stock"]]["owed_buys"]}\n')
            order_time.flush()
            stock_transaction_info[current_row_ta["stock"]]["remaining_buys"] -= \
            stock_transaction_info[current_row_ta["stock"]]["owed_buys"]
            stock_transaction_info[current_row_ta["stock"]]["owed_buys"] = 0
            continue
        elif stock_transaction_info[current_row_ta["stock"]]["owed_sells"] > 0:
            order_time.writelines(
                f'{current_row_ta["stock"]},S,{current_row_ta["idx"]},{stock_transaction_info[current_row_ta["stock"]]["owed_sells"]}\n')
            order_time.flush()
            stock_transaction_info[current_row_ta["stock"]]["remaining_sells"] -= \
            stock_transaction_info[current_row_ta["stock"]]["owed_sells"]
            stock_transaction_info[current_row_ta["stock"]]["owed_sells"] = 0
            continue

    # --- Feature engineering ---

    # --- Add current information to window ---
    stock_ta[current_row_ta["stock"]]["window"].append(current_row_ta["price"])
    stock_ta[current_row_ta["stock"]]["high_window"].append(current_row_ta["high"])
    stock_ta[current_row_ta["stock"]]["low_window"].append(current_row_ta["low"])

    if len(stock_ta[current_row_ta["stock"]]["window"]) > 20:
        stock_ta[current_row_ta["stock"]]["window"].pop(0)
        stock_ta[current_row_ta["stock"]]["high_window"].pop(0)
        stock_ta[current_row_ta["stock"]]["low_window"].pop(0)

    # Hold off on buying and selling for first 15 minutes
    if get_ms(current_row_ta["time"]) < 1800000:
        if (stock_transaction_info[current_row_ta["stock"]]["time_last_bought"] is None) \
                or (((get_ms(current_row_ta["time"]) - stock_transaction_info[current_row_ta["stock"]]["time_last_bought"]) > 60001)
                    and (stock_transaction_info[current_row_ta["stock"]]["buy_transactions"] < 3)):
            stock_transaction_info[current_row_ta["stock"]]["time_last_bought"] = get_ms(current_row_ta["time"])
            stock_transaction_info[current_row_ta["stock"]]["remaining_buys"] -= 1
            stock_transaction_info[current_row_ta["stock"]]["owed_sells"] += 1
            stock_transaction_info[current_row_ta["stock"]]["owed_buys"] -= 1
            stock_transaction_info[current_row_ta["stock"]]["buy_transactions"] += 1
            stock_transaction_info[current_row_ta["stock"]]["current_earnings"] += 1 * current_row_ta["price"]
            order_time.writelines(f'{current_row_ta["stock"]},B,{current_row_ta["idx"]},{1}\n')
            order_time.flush()
            continue
        if (stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] is None) or ((stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] and get_ms(current_row_ta["time"]) -
            stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] > 60001) and (
                stock_transaction_info[current_row_ta["stock"]]["sell_transactions"] < 3)):
            stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] = get_ms(current_row_ta["time"])
            stock_transaction_info[current_row_ta["stock"]]["remaining_sells"] -= 1
            stock_transaction_info[current_row_ta["stock"]]["owed_buys"] += 1
            stock_transaction_info[current_row_ta["stock"]]["owed_sells"] -= 1
            stock_transaction_info[current_row_ta["stock"]]["sell_transactions"] += 1
            stock_transaction_info[current_row_ta["stock"]]["current_earnings"] -= 1 * current_row_ta["price"]

            order_time.writelines(f'{current_row_ta["stock"]},S,{current_row_ta["idx"]},{1}\n')
            order_time.flush()
            continue
        order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
        order_time.flush()
        continue

    # Hold off on buying and selling for last 15 minutes
    if get_ms(current_row_ta["time"]) > 13500000:
        order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
        order_time.flush()
        continue

    # Immediately go to next line if the stock has been bought or sold in the last minute
    if (stock_transaction_info[current_row_ta["stock"]]["time_last_bought"] and
        stock_transaction_info[current_row_ta["stock"]]["time_last_sold"]) \
            and get_ms(current_row_ta["time"]) - stock_transaction_info[current_row_ta["stock"]][
        "time_last_bought"] < 60000 \
            and get_ms(current_row_ta["time"]) - stock_transaction_info[current_row_ta["stock"]][
        "time_last_sold"] < 60000:
        order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
        order_time.flush()
        continue

    if len(stock_ta[current_row_ta["stock"]]["window"]) == 20:
        # --- Calculate bollinger bands ---
        bb_window = stock_ta[current_row_ta["stock"]]["window"]
        ta_bb = ta.volatility.BollingerBands(pd.Series(bb_window), 20, 2)
        current_row_ta["bb_bbhi"] = ta_bb.bollinger_hband_indicator().tolist()[-1]
        current_row_ta["bb_bbli"] = ta_bb.bollinger_lband_indicator().tolist()[-1]

    if len(stock_ta[current_row_ta["stock"]]["window"]) >= 14:
        # --- Calculate RSI ---
        rsi_window = stock_ta[current_row_ta["stock"]]["window"][-14:]
        high_window = stock_ta[current_row_ta["stock"]]["high_window"][-14:]
        low_window = stock_ta[current_row_ta["stock"]]["low_window"][-14:]
        ta_rsi = ta.momentum.RSIIndicator(pd.Series(rsi_window), 14)
        ta_cci = ta.trend.CCIIndicator(close=pd.Series(rsi_window), high=pd.Series(high_window), low=pd.Series(low_window), window=14)
        current_row_ta["rsi"] = ta_rsi.rsi().tolist()[-1]
        current_row_ta["cci"] = ta_cci.cci().tolist()[-1]

    # --- Strategy ---
    # Mean reversion strategy
    action = mean_reversion_strategy(current_row_ta)

    if action == 'N':
        order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
        order_time.flush()
        continue
    elif action == 'B':
        # -- Pass if remaining buys are 0 --
        if stock_transaction_info[current_row_ta["stock"]]["remaining_buys"] == 0:
            order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
            order_time.flush()
            continue

        # -- Pass if previous buy was less than 1 minute ago --
        if stock_transaction_info[current_row_ta["stock"]]["time_last_bought"] and get_ms(current_row_ta["time"]) - \
                stock_transaction_info[current_row_ta["stock"]]["time_last_bought"] < 60001:
            order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
            order_time.flush()
            continue

        weight = 20 - current_row_ta["rsi"]
        vol = random_volume(action, weight, current_row_ta["stock"])

        stock_transaction_info[current_row_ta["stock"]]["time_last_bought"] = get_ms(current_row_ta["time"])
        stock_transaction_info[current_row_ta["stock"]]["remaining_buys"] -= vol
        stock_transaction_info[current_row_ta["stock"]]["owed_sells"] += vol
        stock_transaction_info[current_row_ta["stock"]]["owed_buys"] -= vol
        stock_transaction_info[current_row_ta["stock"]]["buy_transactions"] += 1
        stock_transaction_info[current_row_ta["stock"]]["current_earnings"] += vol * current_row_ta["price"]

        order_time.writelines(f'{current_row_ta["stock"]},B,{current_row_ta["idx"]},{vol}\n')
        order_time.flush()
        continue
    elif action == 'S':
        # -- Pass if remaining sells are 0 --
        if stock_transaction_info[current_row_ta["stock"]]["remaining_sells"] == 0:
            order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
            order_time.flush()
            continue

        # -- Pass if previous sell was less than 1 minute ago --
        if stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] and get_ms(current_row_ta["time"]) - \
                stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] < 60001:
            order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
            order_time.flush()
            continue

        weight = 20 - (100 - current_row_ta["rsi"])
        vol = random_volume(action, weight, current_row_ta["stock"])

        stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] = get_ms(current_row_ta["time"])
        stock_transaction_info[current_row_ta["stock"]]["remaining_sells"] -= vol
        stock_transaction_info[current_row_ta["stock"]]["owed_buys"] += vol
        stock_transaction_info[current_row_ta["stock"]]["owed_sells"] -= vol
        stock_transaction_info[current_row_ta["stock"]]["sell_transactions"] += 1
        stock_transaction_info[current_row_ta["stock"]]["current_earnings"] -= vol * current_row_ta["price"]

        order_time.writelines(f'{current_row_ta["stock"]},S,{current_row_ta["idx"]},{vol}\n')
        order_time.flush()
        continue

# --- End ---
print('\n', end="")
tick_data.close()
order_time.close()
