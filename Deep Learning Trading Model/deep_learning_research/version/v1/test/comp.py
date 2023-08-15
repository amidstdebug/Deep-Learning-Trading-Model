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

    if row['bb_bbhi'] == 1 and row['rsi'] > 80 and row['cci'] >= 100:
        return 'S'
    elif row['bb_bbli'] == 1 and row['rsi'] < 20 and row['cci'] <= -100:
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
        max_sells = stock_transaction_info[sym]['remaining_sells']
        return np.ma.round(np.random.triangular(
            left=1 - 0.5,
            mode=1 + exp_func(impt) * (max_sells - 1),
            right=max_sells + 0.5,
            size=1)
        ).astype(int)[0]
    elif flag == "B":
        max_buys = stock_transaction_info[sym]['remaining_buys']
        return np.ma.round(np.random.triangular(
            left=1 - 0.5,
            mode=1 + exp_func(impt) * (max_buys - 1),
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
stop_loss = {}

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
    # Initialize stock_transaction_info, stock_ta and stop_loss
    if current_row_ta["stock"] not in stock_transaction_info:
        stock_transaction_info[current_row_ta["stock"]] = {
            "remaining_buys": 100,
            "remaining_sells": 100,
            "owed_buys": 0,
            "owed_sells": 0,
            "time_last_bought": None,
            "lowest_bought": None,
            "highest_bought": None,
            "time_last_sold": None,
            "lowest_sold": None,
            "highest_sold": None,
            "buy_transactions": 0,
            "sell_transactions": 0,
            "current_earnings": 0
        }

    if current_row_ta["stock"] not in stock_ta:
        stock_ta[current_row_ta["stock"]] = {
            "window": [],
            "low_window": [],
            "high_window": []
        }

    if current_row_ta["stock"] not in stop_loss:
        stop_loss[current_row_ta["stock"]] = {
            "short_stop_loss": None,
            "short_stop_loss_volume": None,
            "long_stop_loss": None,
            "long_stop_loss_volume": None
        }

    # Pass if remaining buys and sells are 0
    if stock_transaction_info[current_row_ta["stock"]]["remaining_buys"] == 0 and \
            stock_transaction_info[current_row_ta["stock"]]["remaining_sells"] == 0:
        order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
        order_time.flush()
        continue

    # Finish owed transactions
    if current_ms_time >= 14340000:
        if stock_transaction_info[current_row_ta["stock"]]["remaining_sells"] > \
                stock_transaction_info[current_row_ta["stock"]]["remaining_buys"]:
            difference = stock_transaction_info[current_row_ta["stock"]]["remaining_sells"] - \
                            stock_transaction_info[current_row_ta["stock"]]["remaining_buys"]
            order_time.writelines(
                f'{current_row_ta["stock"]},S,{current_row_ta["idx"]},{difference}\n')
            order_time.flush()
            stock_transaction_info[current_row_ta["stock"]]["remaining_sells"] -= \
                difference
            stock_transaction_info[current_row_ta["stock"]]["owed_sells"] = 0
            continue
        elif stock_transaction_info[current_row_ta["stock"]]["remaining_buys"] > \
                stock_transaction_info[current_row_ta["stock"]]["remaining_sells"]:
            difference = stock_transaction_info[current_row_ta["stock"]]["remaining_buys"] - \
                            stock_transaction_info[current_row_ta["stock"]]["remaining_sells"]
            order_time.writelines(
                f'{current_row_ta["stock"]},B,{current_row_ta["idx"]},{difference}\n')
            order_time.flush()
            stock_transaction_info[current_row_ta["stock"]]["remaining_buys"] -= \
                difference
            stock_transaction_info[current_row_ta["stock"]]["owed_buys"] = 0
            continue

    # Hold off on buying and selling for last 5 minutes
    if get_ms(current_row_ta["time"]) > 14100000:
        order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
        order_time.flush()
        continue

    # Fulfill stop loss orders
    if stop_loss[current_row_ta["stock"]]["short_stop_loss"] is not None and \
            (current_ms_time - stock_transaction_info[current_row_ta["stock"]]["time_last_bought"] > 60000):
        if current_row_ta["price"] >= stop_loss[current_row_ta["stock"]]["short_stop_loss"]:
            if stop_loss[current_row_ta['stock']]['short_stop_loss_volume'] > \
                stock_transaction_info[current_row_ta["stock"]]["remaining_buys"]:
                stop_loss[current_row_ta['stock']]['short_stop_loss_volume'] = \
                    stock_transaction_info[current_row_ta["stock"]]["remaining_buys"]
            if stop_loss[current_row_ta['stock']]['short_stop_loss_volume'] <= 0:
                stop_loss[current_row_ta['stock']]['short_stop_loss_volume'] = None
                stop_loss[current_row_ta['stock']]['short_stop_loss'] = None
                order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
                order_time.flush()
                continue
            stock_transaction_info[current_row_ta["stock"]]["remaining_buys"] -= \
                stop_loss[current_row_ta["stock"]]["short_stop_loss_volume"]
            stock_transaction_info[current_row_ta["stock"]]["buy_transactions"] += 1
            stock_transaction_info[current_row_ta["stock"]]["time_last_bought"] = current_ms_time
            stock_transaction_info[current_row_ta["stock"]]["owed_sells"] += \
                stop_loss[current_row_ta["stock"]]["short_stop_loss_volume"]
            stock_transaction_info[current_row_ta["stock"]]["owed_buys"] -= \
                stop_loss[current_row_ta["stock"]]["short_stop_loss_volume"]
            stock_transaction_info[current_row_ta["stock"]]["current_earnings"] -= \
                stop_loss[current_row_ta["stock"]]["short_stop_loss_volume"] * current_row_ta["price"]

            print(f"Fulfilled short stop loss order, {current_row_ta['stock'], stop_loss[current_row_ta['stock']]['short_stop_loss_volume']}")
            order_time.writelines(
                f'{current_row_ta["stock"]},B,{current_row_ta["idx"]},{stop_loss[current_row_ta["stock"]]["short_stop_loss_volume"]}\n')
            order_time.flush()
            stop_loss[current_row_ta["stock"]]["short_stop_loss"] = None
            stop_loss[current_row_ta["stock"]]["short_stop_loss_volume"] = None
            continue

    if stop_loss[current_row_ta["stock"]]["long_stop_loss"] is not None and \
            (current_ms_time - stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] > 60000):
        if current_row_ta["price"] <= stop_loss[current_row_ta["stock"]]["long_stop_loss"]:
            if stop_loss[current_row_ta['stock']]['long_stop_loss_volume'] > \
                stock_transaction_info[current_row_ta["stock"]]["remaining_sells"]:
                stop_loss[current_row_ta['stock']]['long_stop_loss_volume'] = \
                    stock_transaction_info[current_row_ta["stock"]]["remaining_sells"]
            if stop_loss[current_row_ta['stock']]['long_stop_loss_volume'] <= 0:
                stop_loss[current_row_ta['stock']]['long_stop_loss_volume'] = None
                stop_loss[current_row_ta['stock']]['long_stop_loss'] = None
                order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
                order_time.flush()
                continue
            stock_transaction_info[current_row_ta["stock"]]["remaining_sells"] -= \
                stop_loss[current_row_ta["stock"]]["long_stop_loss_volume"]
            stock_transaction_info[current_row_ta["stock"]]["sell_transactions"] += 1
            stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] = current_ms_time
            stock_transaction_info[current_row_ta["stock"]]["owed_buys"] += \
                stop_loss[current_row_ta["stock"]]["long_stop_loss_volume"]
            stock_transaction_info[current_row_ta["stock"]]["owed_sells"] -= \
                stop_loss[current_row_ta["stock"]]["long_stop_loss_volume"]
            stock_transaction_info[current_row_ta["stock"]]["current_earnings"] += \
                stop_loss[current_row_ta["stock"]]["long_stop_loss_volume"] * current_row_ta["price"]
            print(f"Fulfilled long stop loss order, {current_row_ta['stock'], stop_loss[current_row_ta['stock']]['long_stop_loss_volume']}")
            order_time.writelines(
                f'{current_row_ta["stock"]},S,{current_row_ta["idx"]},{stop_loss[current_row_ta["stock"]]["long_stop_loss_volume"]}\n')
            order_time.flush()
            stop_loss[current_row_ta["stock"]]["long_stop_loss"] = None
            stop_loss[current_row_ta["stock"]]["long_stop_loss_volume"] = None
            continue

    # --- Feature engineering ---

    # --- Add current information to window ---
    stock_ta[current_row_ta["stock"]]["window"].append(current_row_ta["price"])
    stock_ta[current_row_ta["stock"]]["low_window"].append(current_row_ta["low"])
    stock_ta[current_row_ta["stock"]]["high_window"].append(current_row_ta["high"])

    if len(stock_ta[current_row_ta["stock"]]["window"]) > 250:
        stock_ta[current_row_ta["stock"]]["window"].pop(0)
        stock_ta[current_row_ta["stock"]]["low_window"].pop(0)
        stock_ta[current_row_ta["stock"]]["high_window"].pop(0)

    # Hold off on buying and selling for first 5 minutes
    if get_ms(current_row_ta["time"]) < 300000:
        if (stock_transaction_info[current_row_ta["stock"]]["time_last_bought"] is None) \
                or (((get_ms(current_row_ta["time"]) - stock_transaction_info[current_row_ta["stock"]]["time_last_bought"]) > 60001)
                    and (stock_transaction_info[current_row_ta["stock"]]["buy_transactions"] < 3)):
            stock_transaction_info[current_row_ta["stock"]]["time_last_bought"] = get_ms(current_row_ta["time"])
            if stock_transaction_info[current_row_ta["stock"]]["lowest_bought"] is None:
                stock_transaction_info[current_row_ta["stock"]]["lowest_bought"] = current_row_ta["price"]
            else:
                if current_row_ta["price"] < stock_transaction_info[current_row_ta["stock"]]["lowest_bought"]:
                    stock_transaction_info[current_row_ta["stock"]]["lowest_bought"] = current_row_ta["price"]
            stock_transaction_info[current_row_ta["stock"]]["remaining_buys"] -= 1
            stock_transaction_info[current_row_ta["stock"]]["owed_sells"] += 1
            stock_transaction_info[current_row_ta["stock"]]["owed_buys"] -= 1
            stock_transaction_info[current_row_ta["stock"]]["buy_transactions"] += 1
            stock_transaction_info[current_row_ta["stock"]]["current_earnings"] += 1 * current_row_ta["price"]
            order_time.writelines(f'{current_row_ta["stock"]},B,{current_row_ta["idx"]},{1}\n')
            order_time.flush()
            #print("\nBought 1 at {}".format(current_row_ta["time"]))
            continue
        if (stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] is None) or ((get_ms(current_row_ta["time"]) -
            stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] > 60001) and (
                stock_transaction_info[current_row_ta["stock"]]["sell_transactions"] < 3)):
            stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] = get_ms(current_row_ta["time"])
            if stock_transaction_info[current_row_ta["stock"]]["lowest_sold"] is None:
                stock_transaction_info[current_row_ta["stock"]]["lowest_sold"] = current_row_ta["price"]
            else:
                if current_row_ta["price"] < stock_transaction_info[current_row_ta["stock"]]["lowest_sold"]:
                    stock_transaction_info[current_row_ta["stock"]]["lowest_sold"] = current_row_ta["price"]
            stock_transaction_info[current_row_ta["stock"]]["remaining_sells"] -= 1
            stock_transaction_info[current_row_ta["stock"]]["owed_buys"] += 1
            stock_transaction_info[current_row_ta["stock"]]["owed_sells"] -= 1
            stock_transaction_info[current_row_ta["stock"]]["sell_transactions"] += 1
            stock_transaction_info[current_row_ta["stock"]]["current_earnings"] -= 1 * current_row_ta["price"]

            order_time.writelines(f'{current_row_ta["stock"]},S,{current_row_ta["idx"]},{1}\n')
            order_time.flush()
            #print("\nSold 1 at {}".format(current_row_ta['time']))
            continue
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

    if len(stock_ta[current_row_ta["stock"]]["window"]) == 250:
        # --- Calculate bollinger bands ---
        bb_window = stock_ta[current_row_ta["stock"]]["window"][-20:]
        ta_bb = ta.volatility.BollingerBands(pd.Series(bb_window), 20, 2)

        current_row_ta["bb_bbhi"] = ta_bb.bollinger_hband_indicator().tolist()[-1]
        current_row_ta["bb_bbli"] = ta_bb.bollinger_lband_indicator().tolist()[-1]

    if len(stock_ta[current_row_ta["stock"]]["window"]) == 250:
        # --- Calculate RSI ---
        rsi_window = stock_ta[current_row_ta["stock"]]["window"][-25:]
        atr_window = stock_ta[current_row_ta["stock"]]["window"][-50:]
        low_window = stock_ta[current_row_ta["stock"]]["low_window"][-50:]
        high_window = stock_ta[current_row_ta["stock"]]["high_window"][-50:]
        ta_rsi = ta.momentum.RSIIndicator(pd.Series(rsi_window), 25)
        ta_atr = ta.volatility.AverageTrueRange(close=pd.Series(atr_window), high=pd.Series(high_window),
                                                low=pd.Series(low_window), window=50)
        ta_cci = ta.trend.CCIIndicator(close=rsi_window[-14:], high=pd.Series(high_window[-14:]), low=pd.Series(low_window[-14:]), window=14)
        current_row_ta["rsi"] = ta_rsi.rsi().tolist()[-1]
        current_row_ta["atr"] = ta_atr.average_true_range().tolist()[-1]
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

        # -- Pass if the current price is less than the lowest sold price --
        # if stock_transaction_info[current_row_ta["stock"]]["lowest_sold"] and current_row_ta["price"] < \
        #         stock_transaction_info[current_row_ta["stock"]]["lowest_sold"]:
        #     order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
        #     order_time.flush()
        #     continue

        weight = 20 - current_row_ta["rsi"]
        vol = random_volume(action, weight, current_row_ta["stock"])

        if vol is None or vol == 0:
            order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
            order_time.flush()
            continue

        # Set the stop-loss
        if stop_loss[current_row_ta["stock"]]["long_stop_loss"] is None:
            stop_loss[current_row_ta["stock"]]["long_stop_loss"] = current_row_ta["price"] - \
                                                                   (current_row_ta["atr"] * 1.50)
            stop_loss[current_row_ta["stock"]]["long_stop_loss_volume"] = vol
        else:
            if current_row_ta["price"] - (current_row_ta["atr"] * 1.50) > stop_loss[current_row_ta["stock"]][
                "long_stop_loss"]:
                stop_loss[current_row_ta["stock"]]["long_stop_loss"] = current_row_ta["price"] - \
                                                                       (current_row_ta["atr"] * 1.50)
                stop_loss[current_row_ta["stock"]]["long_stop_loss_volume"] += vol

        # Alter the stop-loss
        if stop_loss[current_row_ta["stock"]]["short_stop_loss"] is not None:
            if current_row_ta["price"] < stop_loss[current_row_ta["stock"]]["short_stop_loss"]:
                if vol > stop_loss[current_row_ta["stock"]]["short_stop_loss_volume"]:
                    stop_loss[current_row_ta["stock"]]["short_stop_loss"] = None
                    stop_loss[current_row_ta["stock"]]["short_stop_loss_volume"] = None
                elif vol <= stop_loss[current_row_ta["stock"]]["short_stop_loss_volume"]:
                    stop_loss[current_row_ta["stock"]]["short_stop_loss_volume"] -= vol

        stock_transaction_info[current_row_ta["stock"]]["time_last_bought"] = get_ms(current_row_ta["time"])
        if current_row_ta["price"] < stock_transaction_info[current_row_ta["stock"]]["lowest_bought"]:
            stock_transaction_info[current_row_ta["stock"]]["lowest_bought"] = current_row_ta["price"]
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

        if vol is None or vol == 0:
            order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
            order_time.flush()
            continue


        # Set the stop loss
        if stop_loss[current_row_ta["stock"]]["short_stop_loss"] is None:
            stop_loss[current_row_ta["stock"]]["short_stop_loss"] = current_row_ta["price"] + \
                                                                    (current_row_ta["atr"] * 1.50)
            stop_loss[current_row_ta["stock"]]["short_stop_loss_volume"] = vol
        else:
            if current_row_ta["price"] + (current_row_ta["atr"] * 1.50) < stop_loss[current_row_ta["stock"]][
                "short_stop_loss"]:
                stop_loss[current_row_ta["stock"]]["short_stop_loss"] = current_row_ta["price"] + \
                                                                        (current_row_ta["atr"] * 1.50)
                stop_loss[current_row_ta["stock"]]["short_stop_loss_volume"] += vol

        # Alter the stop loss
        if stop_loss[current_row_ta["stock"]]["long_stop_loss"] is not None:
            if current_row_ta["price"] > stop_loss[current_row_ta["stock"]]["long_stop_loss"]:
                if vol > stop_loss[current_row_ta["stock"]]["long_stop_loss_volume"]:
                    stop_loss[current_row_ta["stock"]]["long_stop_loss"] = None
                    stop_loss[current_row_ta["stock"]]["long_stop_loss_volume"] = None
                elif vol <= stop_loss[current_row_ta["stock"]]["long_stop_loss_volume"]:
                    stop_loss[current_row_ta["stock"]]["long_stop_loss_volume"] -= vol



        stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] = get_ms(current_row_ta["time"])
        if current_row_ta["price"] < stock_transaction_info[current_row_ta["stock"]]["lowest_sold"]:
            stock_transaction_info[current_row_ta["stock"]]["lowest_sold"] = current_row_ta["price"]
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
# Print stop loss record
# for stock in stop_loss:
#     if stop_loss[stock]["long_stop_loss"] is not None:
#         print(f'{stock} long stop loss: {stop_loss[stock]["long_stop_loss"]}, volume: {stop_loss[stock]["long_stop_loss_volume"]}')
#         print(f'{stock} highest price: {max(stock_ta[stock]["high_window"])}')
#         print(f'{stock} lowest price: {min(stock_ta[stock]["low_window"])}')
#     if stop_loss[stock]["short_stop_loss"] is not None:
#         print(f'{stock} short stop loss: {stop_loss[stock]["short_stop_loss"]}, volume: {stop_loss[stock]["short_stop_loss_volume"]}')
#         print(f'{stock} highest price: {max(stock_ta[stock]["high_window"])}')
#         print(f'{stock} lowest price: {min(stock_ta[stock]["low_window"])}')

tick_data.close()
order_time.close()
