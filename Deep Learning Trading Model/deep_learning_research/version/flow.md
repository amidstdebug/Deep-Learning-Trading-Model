# Model I/O

$input -> Absolute path of tickdata file e.g. `./input/tickdata_20220805.csv`
<br> $output -> Absolute path of ordertime file e.g. `./output/ordertime_20220805.csv`

# Model Training
1. Import **multiple** original `tickdata_yyyymmdd.csv` files
2. Use `engineer.py` to engineer and consolidate features for model into `trainDATA.csv`
3. Use `train.py` to train model and save model into `model.pkl`

# Model Testing
1. Import **single** original `tickdata_yyyymmdd.csv` file 
2. ~~Use `engineer.py` to engineer and consolidate features for model into `test_yyyymmdd.csv`~~
3. Use `engineer.py` to engineer and consolidate features for model **on a tick by tick basis**
3. Use `run.sh` to call predict.py to predict ordertime and save ordertime into `ordertime_yyyymmdd.csv` by running `predict.py`
4. Use `evaluate.py` to evaluate model performance
5. Save earnings into `earning_yyyymmdd.txt`

# File Structure
File structure will be as follows
```
v1/
├── train/
│   ├── original_training_files/
│   │   ├── tickdata_20220805.csv
│   │   ├── tickdata_20220807.csv
│   │   └── tickdata_20220808.csv
│   ├── traindata.csv
│   └── train.py
├── test/
│   ├── run.sh
│   ├── predict.py
│   ├── test_20220910/
│   │   ├── tickdata_20220910.csv
│   │   ├── ordertime_20220910.csv
│   │   └── earning_20220910.txt
│   └── test_20221010/
│       ├── tickdata_20221010.csv
│       ├── ordertime_20221010.csv
│       └── earning_20221010.txt
├── engineer.py
└── model.pkl
```
