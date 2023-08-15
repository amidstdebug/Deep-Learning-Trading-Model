import math
from collections import deque
import random
import numpy as np
import pandas as pd
import sys
import ta
import time
import tensorflow as tf
from datetime import datetime

tick_data_path = sys.argv[1]
order_time_path = sys.argv[2]
start_time = time.time()
stock_list = pd.read_csv('stock_list.csv')

CONV_WIDTH = 3
model = tf.keras.models.load_model('nn_model')


# -- Window Class --
class WindowGenerator:
    def __init__(self, input_width, label_width, shift, trainStocksList,
                 label_columns=None):
        # Store the raw data.
        self.trainStocksList = trainStocksList

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(trainStocksList[0].columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        trainStockDSList = []

        for stock in self.trainStocksList:
            trainStockDSList.append(self.make_dataset(stock))

        return trainStockDSList


# --- Functions ---
def get_ms(tm):
    hhmmss = tm // 1000
    ms = (hhmmss // 10000 * 3600 + (hhmmss // 100 % 100) * 60 + hhmmss % 100) * 1000 + tm % 1000
    ms_from_open = ms - 34200000  # millisecond from stock opening
    if tm >= 130000000:
        ms_from_open -= 5400000
    return ms_from_open


def timeseries_dataset_from_array(
        data,
        targets,
        sequence_length,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=128,
        shuffle=False,
        seed=None,
        start_index=None,
        end_index=None,
):
    """Creates a dataset of sliding windows over a timeseries provided as array.
    This function takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    length of the sequences/windows, spacing between two sequence/windows, etc.,
    to produce batches of timeseries inputs and targets.
    Args:
      data: Numpy array or eager tensor
        containing consecutive data points (timesteps).
        Axis 0 is expected to be the time dimension.
      targets: Targets corresponding to timesteps in `data`.
        `targets[i]` should be the target
        corresponding to the window that starts at index `i`
        (see example 2 below).
        Pass None if you don't have target data (in this case the dataset will
        only yield the input data).
      sequence_length: Length of the output sequences (in number of timesteps).
      sequence_stride: Period between successive output sequences.
        For stride `s`, output samples would
        start at index `data[i]`, `data[i + s]`, `data[i + 2 * s]`, etc.
      sampling_rate: Period between successive individual timesteps
        within sequences. For rate `r`, timesteps
        `data[i], data[i + r], ... data[i + sequence_length]`
        are used for creating a sample sequence.
      batch_size: Number of timeseries samples in each batch
        (except maybe the last one). If `None`, the data will not be batched
        (the dataset will yield individual samples).
      shuffle: Whether to shuffle output samples,
        or instead draw them in chronological order.
      seed: Optional int; random seed for shuffling.
      start_index: Optional int; data points earlier (exclusive)
        than `start_index` will not be used
        in the output sequences. This is useful to reserve part of the
        data for test or validation.
      end_index: Optional int; data points later (exclusive) than `end_index`
        will not be used in the output sequences.
        This is useful to reserve part of the data for test or validation.
    Returns:
      A tf.data.Dataset instance. If `targets` was passed, the dataset yields
      tuple `(batch_of_sequences, batch_of_targets)`. If not, the dataset yields
      only `batch_of_sequences`.
    Example 1:
    Consider indices `[0, 1, ... 99]`.
    With `sequence_length=10,  sampling_rate=2, sequence_stride=3`,
    `shuffle=False`, the dataset will yield batches of sequences
    composed of the following indices:
    ```
    First sequence:  [0  2  4  6  8 10 12 14 16 18]
    Second sequence: [3  5  7  9 11 13 15 17 19 21]
    Third sequence:  [6  8 10 12 14 16 18 20 22 24]
    ...
    Last sequence:   [78 80 82 84 86 88 90 92 94 96]
    ```
    In this case the last 3 data points are discarded since no full sequence
    can be generated to include them (the next sequence would have started
    at index 81, and thus its last step would have gone over 99).
    Example 2: Temporal regression.
    Consider an array `data` of scalar values, of shape `(steps,)`.
    To generate a dataset that uses the past 10
    timesteps to predict the next timestep, you would use:
    ```python
    input_data = data[:-10]
    targets = data[10:]
    dataset = tf.keras.utils.timeseries_dataset_from_array(
        input_data, targets, sequence_length=10)
    for batch in dataset:
      inputs, targets = batch
      assert np.array_equal(inputs[0], data[:10])  # First sequence: steps [0-9]
      # Corresponding target: step 10
      assert np.array_equal(targets[0], data[10])
      break
    ```
    Example 3: Temporal regression for many-to-many architectures.
    Consider two arrays of scalar values `X` and `Y`,
    both of shape `(100,)`. The resulting dataset should consist samples with
    20 timestamps each. The samples should not overlap.
    To generate a dataset that uses the current timestamp
    to predict the corresponding target timestep, you would use:
    ```python
    X = np.arange(100)
    Y = X*2
    sample_length = 20
    input_dataset = tf.keras.utils.timeseries_dataset_from_array(
      X, None, sequence_length=sample_length, sequence_stride=sample_length)
    target_dataset = tf.keras.utils.timeseries_dataset_from_array(
      Y, None, sequence_length=sample_length, sequence_stride=sample_length)
    for batch in zip(input_dataset, target_dataset):
      inputs, targets = batch
      assert np.array_equal(inputs[0], X[:sample_length])
      # second sample equals output timestamps 20-40
      assert np.array_equal(targets[1], Y[sample_length:2*sample_length])
      break
    ```
    """
    if start_index:
        if start_index < 0:
            raise ValueError(
                "`start_index` must be 0 or greater. Received: "
                f"start_index={start_index}"
            )
        if start_index >= len(data):
            raise ValueError(
                "`start_index` must be lower than the length of the "
                f"data. Received: start_index={start_index}, for data "
                f"of length {len(data)}"
            )
    if end_index:
        if start_index and end_index <= start_index:
            raise ValueError(
                "`end_index` must be higher than `start_index`. "
                f"Received: start_index={start_index}, and "
                f"end_index={end_index} "
            )
        if end_index >= len(data):
            raise ValueError(
                "`end_index` must be lower than the length of the "
                f"data. Received: end_index={end_index}, for data of "
                f"length {len(data)}"
            )
        if end_index <= 0:
            raise ValueError(
                "`end_index` must be higher than 0. "
                f"Received: end_index={end_index}"
            )

    # Validate strides
    if sampling_rate <= 0:
        raise ValueError(
            "`sampling_rate` must be higher than 0. Received: "
            f"sampling_rate={sampling_rate}"
        )
    if sampling_rate >= len(data):
        raise ValueError(
            "`sampling_rate` must be lower than the length of the "
            f"data. Received: sampling_rate={sampling_rate}, for data "
            f"of length {len(data)}"
        )
    if sequence_stride <= 0:
        raise ValueError(
            "`sequence_stride` must be higher than 0. Received: "
            f"sequence_stride={sequence_stride}"
        )
    if sequence_stride >= len(data):
        raise ValueError(
            "`sequence_stride` must be lower than the length of the "
            f"data. Received: sequence_stride={sequence_stride}, for "
            f"data of length {len(data)}"
        )

    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(data)

    # Determine the lowest dtype to store start positions (to lower memory
    # usage).
    num_seqs = end_index - start_index - (sequence_length * sampling_rate) + 1
    if targets is not None:
        num_seqs = min(num_seqs, len(targets))
    if num_seqs < 2147483647:
        index_dtype = "int32"
    else:
        index_dtype = "int64"

    # Generate start positions
    start_positions = np.arange(0, num_seqs, sequence_stride, dtype=index_dtype)
    if shuffle:
        if seed is None:
            seed = np.random.randint(1e6)
        rng = np.random.RandomState(seed)
        rng.shuffle(start_positions)

    sequence_length = tf.cast(sequence_length, dtype=index_dtype)
    sampling_rate = tf.cast(sampling_rate, dtype=index_dtype)

    positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat()

    # For each initial window position, generates indices of the window elements
    indices = tf.data.Dataset.zip(
        (tf.data.Dataset.range(len(start_positions)), positions_ds)
    ).map(
        lambda i, positions: tf.range(
            positions[i],
            positions[i] + sequence_length * sampling_rate,
            sampling_rate,
        ),

    )

    dataset = sequences_from_indices(data, indices, start_index, end_index)
    if targets is not None:
        indices = tf.data.Dataset.zip(
            (tf.data.Dataset.range(len(start_positions)), positions_ds)
        ).map(
            lambda i, positions: positions[i],

        )
        target_ds = sequences_from_indices(
            targets, indices, start_index, end_index
        )
        dataset = tf.data.Dataset.zip((dataset, target_ds))
    if batch_size is not None:
        if shuffle:
            # Shuffle locally at each iteration
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        dataset = dataset.batch(batch_size)
    else:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024, seed=seed)
    return dataset


def sequences_from_indices(array, indices_ds, start_index, end_index):
    dataset = tf.data.Dataset.from_tensors(array[start_index:end_index])
    dataset = tf.data.Dataset.zip((dataset.repeat(), indices_ds)).map(
        lambda steps, inds: tf.gather(steps, inds),

    )
    return dataset


def sell(volume, row_info, order_file):
    stock_transaction_info[row_info["stock"]]["time_last_sold"] = get_ms(row_info["time"])
    stock_transaction_info[row_info["stock"]]["remaining_sells"] -= volume
    stock_transaction_info[row_info["stock"]]["owed_buys"] += volume
    stock_transaction_info[row_info["stock"]]["owed_sells"] -= volume
    stock_transaction_info[row_info["stock"]]["sell_transactions"] += 1
    stock_transaction_info[row_info["stock"]]["current_earnings"] -= volume * row_info["price"]

    order_file.writelines(f'{row_info["stock"]},S,{row_info["idx"]},{volume}\n')
    order_file.flush()
    return ""


def buy(volume, row_info, order_file):
    stock_transaction_info[row_info["stock"]]["time_last_bought"] = get_ms(row_info["time"])
    stock_transaction_info[row_info["stock"]]["remaining_buys"] -= volume
    stock_transaction_info[row_info["stock"]]["owed_sells"] += volume
    stock_transaction_info[row_info["stock"]]["owed_buys"] -= volume
    stock_transaction_info[row_info["stock"]]["buy_transactions"] += 1
    stock_transaction_info[row_info["stock"]]["current_earnings"] += volume * row_info["price"]

    order_file.writelines(f'{row_info["stock"]},B,{row_info["idx"]},{volume}\n')
    order_file.flush()
    return ""


def calculate_skewness(data):
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data)

    if std_dev == 0:
        return 'No Purchase'

    if not np.isfinite(std_dev):
        return None

    skewness = (1 / n) * sum(((x - mean) / std_dev) ** 3 for x in data)
    return skewness


stock_difference = set()
percentage_stock_difference = set()

def nn_strategy(row):
    """
	Define a function to determine if the stock is overbought or oversold based on its position relative
	to the output of neural network and current market conditions

	:param row:
	:param model: neural network model
	:param weights: array of weights for technical indicators
	:return: 'B' if the stock is overbought,'S' if the stock is oversold,'N' if the stock is neither
	"""

    # Run neural network model to get prediction
    # model_output = model.predict(model_input)
    if predictions[row['stock']] is None or stock_transaction_info[row['stock']]['ticks_elapsed'] % 20 == 0:
        # Predict the price using the model
        dict_copy = stock_ta[row['stock']].copy()
        del dict_copy['window']
        del dict_copy['high_window']
        del dict_copy['low_window']
        one_hot_sym = pd.get_dummies(
            stock_list['Stocks'].tolist() + [*[row['stock']] * len(stock_ta[row['stock']]['COLUMN07'])]).iloc[
                      500:].reset_index(drop=True)
        model_input = pd.DataFrame(dict_copy)
        model_window = WindowGenerator(
            input_width=18,
            label_width=1,
            shift=1,
            trainStocksList=[pd.concat([model_input, one_hot_sym], axis=1)],
            label_columns=['COLUMN07']
        )
        predictions[row['stock']] = model_output = model.predict(model_window.train[0], verbose=0)[0, :, 0]
        #print('model output length:',len(model_output))
        model_output = model_output[0]
        difference = abs(np.mean(np.array([*[row['price']] * 100]) - np.array(predictions[row['stock']])))
        percentage_difference = abs((np.mean(np.array([*[row['price']] * 100]) - np.array(predictions[row['stock']])) / row['price']) * 100)
        if difference >= 100:
            stock_difference.add(row['stock'])
        if percentage_difference >= 20:
            percentage_stock_difference.add(row['stock'])
        #print(f'\nNumber of stocks with difference greater than 100: {len(stock_difference)}')
        #print(f'\nNumber of stocks with percentage difference greater than 20: {len(percentage_stock_difference)}')

    # take difference of low and high price and multiply it by a random factor between -0.5 to 0.5 and add it to the price
    # model_output = row['price'] + (row['high_price'] - row['low_price']) * (random.random() - 0.5)

        up_down = model_output - row['price']
        model_confidence_buy = (row['high_price'] - model_output) / row['high_price'] * 100
        model_confidence_sell = (row['low_price'] - model_output) / row['low_price'] * 100
    
        selling_volume_skew_fn = calculate_skewness(row['selling_volume_skew'])
        buying_volume_skew_fn = calculate_skewness(row['buying_volume_skew'])
        selling_volume_average_fn = row['selling_volume_average']
        buying_volume_average_fn = row['buying_volume_average']
    
        max_risk = (row['price'] - row['high_price']) / row['price'] * 100
        min_risk = (row['price'] - row['low_price']) / row['price'] * 100
    
        if selling_volume_skew_fn == 'No Purchase' or buying_volume_skew_fn == 'No Purchase':
            tradeable_volume = 0
        else:
            tradeable_volume = (min(buying_volume_average_fn, selling_volume_average_fn) * (
                    1 - max(abs(model_confidence_buy), abs(model_confidence_sell)) / 100) * (1 - max_risk / 100) * (
                                        1 + min_risk / 100) * (
                                        1 + selling_volume_skew_fn + buying_volume_skew_fn)) / 10000
            if tradeable_volume < 1:
                tradeable_volume = 0
                
            tradeable_volume = math.ceil(tradeable_volume)
            tradeable_volume = min(tradeable_volume, 10)
    
            # if tradeable_volume > 100:
            #     print("\n", tradeable_volume)
                
        if percentage_difference < 20:
            # --- If model performance is decent enough ---
            if up_down > 0:  # longing
                if model_output > row['high_price']:  # high risk
                    return 'B', min(tradeable_volume, 5)  # cap it at 5 shares to reduce risk
                else:
                    return 'B', tradeable_volume
            elif up_down < 0:  # shorting
                if model_output < row['low_price']:  # high risk
                    return 'S', min(tradeable_volume, 5)  # cap it at 5 shares to reduce risk
                else:
                    return 'S', tradeable_volume
        else:
            # --- If model performance is bad ---
            if row['bb_bbhi'] == 1 and row['RSI'] > 80:
                return 'S', tradeable_volume
            elif row['bb_bbli'] == 1 and row['RSI'] < 20:
                return 'B', tradeable_volume
            else:
                return 'N', 0
    else:
        # --- run within 20 tick window when NN is not being run---
        stock_symbol = row['stock']
        if row['bb_bbhi'] == 1 and row['RSI'] > 80:
            weight = 20 - (100 - row["RSI"])
            RMS_tradeable_volume = random_volume('S', weight, stock_symbol)
            return 'S', RMS_tradeable_volume
        elif row['bb_bbli'] == 1 and row['RSI'] < 20:
            weight = 20 - row["RSI"]
            RMB_tradeable_volume = random_volume('B', weight, stock_symbol)
            return 'B', RMB_tradeable_volume
        else:
            return 'N', 0


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
        if max_sells <= 0:
            return 0
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
        if max_buys <= 0:
            return 0

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
predictions = {}

# --- Loop through each line in tick_data ---
while True:
    # --- Perform feature engineering per stock ---
    current_line = tick_data.readline()
    # print(current_line)
    # Break condition
    if current_line.strip() == '' or len(current_line) == 0:
        break

    current_row = current_line.split(',')

    # Initialize stock_transaction_info and stock_ta
    if current_row[1] not in stock_transaction_info:
        stock_transaction_info[current_row[1]] = {
            "remaining_buys": 100,
            "remaining_sells": 100,
            "owed_buys": 0,
            "owed_sells": 0,
            "time_last_bought": None,
            "time_last_sold": None,
            "buy_transactions": 0,
            "sell_transactions": 0,
            "current_earnings": 0,
            "ticks_elapsed": 0,
        }

    if current_row[1] not in stock_ta:
        stock_ta[current_row[1]] = {
            "window": [],
            "high_window": [],
            "low_window": [],
            "COLUMN07": [],
            "bb_bbm": [],
            "bb_bbh": [],
            "bb_bbl": [],
            "bb_bbhi": [],
            "bb_bbli": [],
            "RSI": [],
            "MACD": [],
            "stoch_k": [],
            "stoch_d": [],
            "ATR": [],
            "RMA": []
        }

    if current_row[1] not in predictions:
        predictions[current_row[1]] = None

    # Pass if remaining buys and sells are 0
    if stock_transaction_info[current_row[1]]["remaining_buys"] == 0 and \
            stock_transaction_info[current_row[1]]["remaining_sells"] == 0:
        order_time.writelines(f'{current_row[1]},N,{current_row[0]},0\n')
        order_time.flush()
        continue

    selling_volume_skew = [int(x) for x in current_row[17:27]]
    buying_volume_skew = [int(x) for x in current_row[37:47]]

    selling_volume_average = np.mean(np.array([int(x) for x in current_row[17:27]]))
    buying_volume_average = np.mean(np.array([int(x) for x in current_row[37:47]]))

    current_row_ta = {
        "idx": current_row[0],
        "stock": current_row[1],
        "time": int(current_row[2]),
        "high": float(current_row[4]),
        "low": float(current_row[5]),
        "price": float(current_row[6]),
        "rsi": None,
        "bb_bbhi": None,
        "bb_bbli": None,
        "high_price": float(current_row[4]),
        "low_price": float(current_row[5]),
        "selling_volume_skew": selling_volume_skew,
        "buying_volume_skew": buying_volume_skew,
        "selling_volume_average": selling_volume_average,
        "buying_volume_average": buying_volume_average,
        "weighted_buying": float(current_row[53]),
        "weighted_selling": float(current_row[54])
    }
    current_ms_time = get_ms(int(current_row[2]))

    print(
        f'Index: \x1b[34m{current_row_ta["idx"]}\x1b[0m \x1b[0m(\x1b[32m{(current_ms_time / 14400000) * 100:.2f}\x1b[0m%), Time: \x1b[33m{(time.time() - start_time):.2f}s\x1b[0m',
        end='\r'
    )

    stock_transaction_info[current_row_ta["stock"]]["ticks_elapsed"] += 1

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

    if len(stock_ta[current_row_ta["stock"]]["window"]) > 30:
        stock_ta[current_row_ta["stock"]]["window"].pop(0)
        stock_ta[current_row_ta["stock"]]["high_window"].pop(0)
        stock_ta[current_row_ta["stock"]]["low_window"].pop(0)

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

    # Start feature engineering after 29 minutes
    if get_ms(current_row_ta["time"]) > 1740000:
        if len(stock_ta[current_row_ta["stock"]]["window"]) == 30:
            fe_start = time.time()
            stock_ta[current_row_ta["stock"]]["COLUMN07"].append(current_row_ta["price"])
            macd_window = stock_ta[current_row_ta["stock"]]["window"]
            ta_macd = ta.trend.MACD(pd.Series(macd_window), window_slow=26, window_fast=12, window_sign=9)
            current_row_ta["MACD"] = ta_macd.macd().tolist()[-1]
            stock_ta[current_row_ta["stock"]]["MACD"].append(current_row_ta["MACD"])

            # --- Calculate bollinger bands ---
            bb_window = stock_ta[current_row_ta["stock"]]["window"][-20:]
            ta_bb = ta.volatility.BollingerBands(pd.Series(bb_window), 20, 2)
            current_row_ta["bb_bbhi"] = ta_bb.bollinger_hband_indicator().tolist()[-1]
            stock_ta[current_row_ta["stock"]]["bb_bbhi"].append(current_row_ta["bb_bbhi"])
            current_row_ta["bb_bbli"] = ta_bb.bollinger_lband_indicator().tolist()[-1]
            stock_ta[current_row_ta["stock"]]["bb_bbli"].append(current_row_ta["bb_bbli"])
            current_row_ta["bb_bbm"] = ta_bb.bollinger_mavg().tolist()[-1]
            stock_ta[current_row_ta["stock"]]["bb_bbm"].append(current_row_ta["bb_bbm"])
            current_row_ta["bb_bbh"] = ta_bb.bollinger_hband().tolist()[-1]
            stock_ta[current_row_ta["stock"]]["bb_bbh"].append(current_row_ta["bb_bbh"])
            current_row_ta["bb_bbl"] = ta_bb.bollinger_lband().tolist()[-1]
            stock_ta[current_row_ta["stock"]]["bb_bbl"].append(current_row_ta["bb_bbl"])

            # --- Calculate RSI ---
            rsi_window = stock_ta[current_row_ta["stock"]]["window"]
            low_window = stock_ta[current_row_ta["stock"]]["low_window"]
            high_window = stock_ta[current_row_ta["stock"]]["high_window"]
            ta_rsi = ta.momentum.RSIIndicator(pd.Series(rsi_window), 14)
            ta_atr = ta.volatility.AverageTrueRange(close=pd.Series(rsi_window), high=pd.Series(high_window),
                                                    low=pd.Series(low_window), window=14)
            ta_stoch = ta.momentum.StochasticOscillator(high=pd.Series(high_window), low=pd.Series(low_window),
                                                        close=pd.Series(rsi_window), window=14, smooth_window=3)
            current_row_ta["RSI"] = ta_rsi.rsi().tolist()[-1]
            stock_ta[current_row_ta["stock"]]["RSI"].append(current_row_ta["RSI"])
            current_row_ta["ATR"] = ta_atr.average_true_range().tolist()[-1]
            stock_ta[current_row_ta["stock"]]["ATR"].append(current_row_ta["ATR"])
            current_row_ta["stoch_k"] = ta_stoch.stoch().tolist()[-1]
            stock_ta[current_row_ta["stock"]]["stoch_k"].append(current_row_ta["stoch_k"])
            current_row_ta["stoch_d"] = ta_stoch.stoch_signal().tolist()[-1]
            stock_ta[current_row_ta["stock"]]["stoch_d"].append(current_row_ta["stoch_d"])

            # --- Calculate RMA ---
            rma_window = stock_ta[current_row_ta["stock"]]["window"][-10:]
            ta_rma = ta.trend.EMAIndicator(pd.Series(rma_window), 10)
            current_row_ta["RMA"] = ta_rma.ema_indicator().tolist()[-1]
            stock_ta[current_row_ta["stock"]]["RMA"].append(current_row_ta["RMA"])

            # print('')
            # for key in stock_ta[current_row_ta["stock"]].keys():
            #     print(f'{key}:\x1b[34m{len(stock_ta[current_row_ta["stock"]][key])}\x1b[0m ', end="")
            # print('')

    # Hold off on buying and selling for first 30 minutes
    if get_ms(current_row_ta["time"]) < 1800000:
        if (not stock_transaction_info[current_row_ta["stock"]]["time_last_bought"]) or ((stock_transaction_info[
                                                                                              current_row_ta["stock"]][
                                                                                              "time_last_bought"] and get_ms(
                current_row_ta["time"]) - \
                                                                                          stock_transaction_info[
                                                                                              current_row_ta["stock"]][
                                                                                              "time_last_bought"] > 60001) and (
                                                                                                 stock_transaction_info[
                                                                                                     current_row_ta[
                                                                                                         "stock"]][
                                                                                                     "buy_transactions"] < 3)):
            buy(1, current_row_ta, order_time)
            continue
        if (not stock_transaction_info[current_row_ta["stock"]]["time_last_sold"]) or (
                (stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] and get_ms(current_row_ta["time"]) -
                 stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] > 60001) and (
                        stock_transaction_info[current_row_ta["stock"]]["sell_transactions"] < 3)):
            sell(1, current_row_ta, order_time)
            continue
        order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
        order_time.flush()
        continue

    # Hold off on buying and selling for last 15 minutes
    if get_ms(current_row_ta["time"]) > 13500000:
        order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
        order_time.flush()
        continue

    # --- Strategy ---
    # Mean Reversion Strategy

    # Neural Network Strategy
    action, vol = nn_strategy(current_row_ta)
    if vol > 100:
        print("\n", vol, "\n")
    if action == 'N' or vol == 0 or vol > 97:
        order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
        order_time.flush()
        continue
    elif action == 'B':
        # -- Pass if remaining buys are 0 --
        if stock_transaction_info[current_row_ta["stock"]]["remaining_buys"] == 0:
            order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
            order_time.flush()
            continue

        if vol > stock_transaction_info[current_row_ta["stock"]]["remaining_buys"]:
            vol = stock_transaction_info[current_row_ta["stock"]]["remaining_buys"]

        # -- Pass if previous buy was less than 1 minute ago --
        if stock_transaction_info[current_row_ta["stock"]]["time_last_bought"] and get_ms(current_row_ta["time"]) - \
                stock_transaction_info[current_row_ta["stock"]]["time_last_bought"] < 60001:
            order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
            order_time.flush()
            continue
        #
        # weight = 20 - current_row_ta["RSI"]
        # vol = random_volume(action, weight, current_row_ta["stock"])

        buy(vol, current_row_ta, order_time)
        continue
    elif action == 'S':
        # -- Pass if remaining sells are 0 --
        if stock_transaction_info[current_row_ta["stock"]]["remaining_sells"] == 0:
            order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
            order_time.flush()
            continue

        if vol > stock_transaction_info[current_row_ta["stock"]]["remaining_sells"]:
            vol = stock_transaction_info[current_row_ta["stock"]]["remaining_sells"]

        # -- Pass if previous sell was less than 1 minute ago --
        if stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] and get_ms(current_row_ta["time"]) - \
                stock_transaction_info[current_row_ta["stock"]]["time_last_sold"] < 60001:
            order_time.writelines(f'{current_row_ta["stock"]},N,{current_row_ta["idx"]},0\n')
            order_time.flush()
            continue

        sell(vol, current_row_ta, order_time)
        continue

# --- End ---
print('\n')
tick_data.close()
order_time.close()
