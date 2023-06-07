#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime

data = pd.read_csv('prepared_data_au.csv.gz')
data['date'] = data['date'].astype('int').astype('string')

# data[data['date']<='20230131'].to_csv('train_data_au.csv.gz',index=False)
# data[data['date']>'20230131'].to_csv('test_data_au.csv.gz',index=False)


data['signal_long'] = (data['bid_1'].shift(-360) - data['ask_1'].shift(-2))*1000
data['signal_short'] = (-data['ask_1'].shift(-360) + data['bid_1'].shift(-2))*1000

data['signal_long'] = data['signal_long'].apply(lambda x:1 if x > 15 else 0)
data['signal_short'] = data['signal_short'].apply(lambda x:-1 if x > 15 else 0)

data['signal'] = data['signal_long'] + data['signal_short']
data['signal'].value_counts()

data[['time','signal']].to_csv('label.csv.gz',index=False)



