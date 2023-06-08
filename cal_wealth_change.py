import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime

target_name = '1_min_tp4_sl4_10yuan_1delay_target'

data = pd.read_csv('data/data_night_shifted_au.csv.gz')
data = data.set_index('time')
data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S.%f')

period = 2*60*1
delay_period = 1
sl_ratio = 4
tp_ratio = 4

potential_list = []
wealth_factor = {'long': [], 'short': []}

for i, (d, df) in enumerate(data.groupby('date')):
    print(f"{d}", end='\r')
    # if i < 60:
    #     continue
    df = df[~df.index.duplicated()]
    night_hour = (df.index.time >= datetime.time(21,5,0)) | (df.index.time < datetime.time(2,25,0))
    morning_hour = (df.index.time >= datetime.time(9,5,0)) & (df.index.time < datetime.time(11,25,0))
    afternoon_hour = (df.index.time >= datetime.time(13,35,0)) & (df.index.time < datetime.time(14,55,0))
    trading_idx = df.index[night_hour | morning_hour | afternoon_hour]
    bid = df['bid_1']
    ask = df['ask_1']
    full_half_sec = pd.date_range(df.index.min(), df.index.max(), freq=pd.Timedelta(0.5, 's'))
    # break

    filled_bid = bid[~bid.index.duplicated()].reindex(full_half_sec).interpolate(method='pad', limit=period, limit_direction='forward').dropna()
    filled_ask = ask[~ask.index.duplicated()].reindex(full_half_sec).interpolate(method='pad', limit=period, limit_direction='forward').dropna()
    spread = (ask/bid) - 1
    spread = spread.rolling(50).mean() + (10/1000)/((bid + ask)/2) #10 yuan per lot
    # spread = spread.clip(lower=) #10 yuan per lot
    spread = spread.reindex(trading_idx)
    sl_short = 1/(1+sl_ratio*spread)
    sl_long = 1-sl_ratio*spread
    tp_short = 1/(1-tp_ratio*spread)
    tp_long = 1+tp_ratio*spread
    long_s = []
    short_s = []

    for i in range(1+delay_period, period+1+delay_period):
        short_s.append(2 - (filled_ask.shift(-i).reindex(trading_idx)/filled_bid.shift(-delay_period).reindex(trading_idx)))
        long_s.append((filled_bid.shift(-i).reindex(trading_idx)/filled_ask.shift(-delay_period).reindex(trading_idx)))
    
    short_pnl = pd.concat(short_s, axis=1, ignore_index=True)
    long_pnl = pd.concat(long_s, axis=1, ignore_index=True)

    short_tp_breach = np.argmax((short_pnl >= tp_short.values[:,np.newaxis]), axis=1)
    short_tp_breach[short_tp_breach == 0] = short_pnl.shape[1]
    short_sl_breach = np.argmax((short_pnl <= sl_short.values[:,np.newaxis]), axis=1)
    short_sl_breach[short_sl_breach == 0] = short_pnl.shape[1]

    long_tp_breach = np.argmax((long_pnl >= tp_short.values[:,np.newaxis]), axis=1)
    long_tp_breach[long_tp_breach == 0] = long_pnl.shape[1]
    long_sl_breach = np.argmax((long_pnl <= sl_short.values[:,np.newaxis]), axis=1)
    long_sl_breach[long_sl_breach == 0] = long_pnl.shape[1]

    short_tp_trig = (short_tp_breach < short_sl_breach)
    short_sl_trig = (short_sl_breach < short_tp_breach)
    long_tp_trig = (long_tp_breach < long_sl_breach)
    long_sl_trig = (long_sl_breach < long_tp_breach)

    short_tp_breach = np.argmax((short_pnl >= tp_short.values[:,np.newaxis]), axis=1)
    short_tp_breach[short_tp_breach == 0] = short_pnl.shape[1]
    short_sl_breach = np.argmax((short_pnl <= sl_short.values[:,np.newaxis]), axis=1)
    short_sl_breach[short_sl_breach == 0] = short_pnl.shape[1]

    long_tp_breach = np.argmax((long_pnl >= tp_short.values[:,np.newaxis]), axis=1)
    long_tp_breach[long_tp_breach == 0] = long_pnl.shape[1]
    long_sl_breach = np.argmax((long_pnl <= sl_short.values[:,np.newaxis]), axis=1)
    long_sl_breach[long_sl_breach == 0] = long_pnl.shape[1]

    short_tp_trig = (short_tp_breach < short_sl_breach)
    short_sl_trig = (short_sl_breach < short_tp_breach)
    long_tp_trig = (long_tp_breach < long_sl_breach)
    long_sl_trig = (long_sl_breach < long_tp_breach)

    # print(f"ltp={long_tp_trig.sum()} lsl={long_sl_trig.sum()} stp={short_tp_trig.sum()} ssl={short_sl_trig.sum()}")

    short_wealth_factor = short_pnl.iloc[:,-1].copy()
    # short
    if short_tp_trig.sum() > 0:
        idx = np.vstack([np.arange(short_tp_trig.sum()), short_tp_breach[short_tp_trig]])
        short_wealth_factor[short_tp_trig] = short_pnl.loc[short_tp_trig].values[tuple(idx)]

    if short_sl_trig.sum() > 0:
        idx = np.vstack([np.arange(short_sl_trig.sum()), short_sl_breach[short_sl_trig]])
        short_wealth_factor[short_sl_trig] = short_pnl.loc[short_sl_trig].values[tuple(idx)]

    long_wealth_factor = long_pnl.iloc[:,-1].copy()
    # long
    if long_tp_trig.sum() > 0:
        idx = np.vstack([np.arange(long_tp_trig.sum()), long_tp_breach[long_tp_trig]])
        long_wealth_factor[long_tp_trig] = long_pnl.loc[long_tp_trig].values[tuple(idx)]

    if long_sl_trig.sum() > 0:
        idx = np.vstack([np.arange(long_sl_trig.sum()), long_sl_breach[long_sl_trig]])
        long_wealth_factor[long_sl_trig] = long_pnl.loc[long_sl_trig].values[tuple(idx)]

    wealth_factor['short'].append(short_wealth_factor.copy())
    wealth_factor['long'].append(long_wealth_factor.copy())

    del long_pnl
    del short_pnl

    long_potential = (long_wealth_factor > 1).mean()
    short_potential = (short_wealth_factor > 1).mean()

    long_mean_profit = long_wealth_factor[long_wealth_factor > 1].mean()
    short_mean_profit = short_wealth_factor[short_wealth_factor > 1].mean()

    # print(f"{d}: {long_potential=:.5f}, {long_mean_profit=:.5f}, {short_potential=:.5f}, {short_mean_profit=:.5f}")
    potential_list.append({'date': d, 'count': trading_idx.shape[0], 'long_signal_pct': long_potential, 'long_mean_profit': long_mean_profit,
                           'short_signal_pct': short_potential, 'short_mean_profit': short_mean_profit})
    
pot_df = pd.DataFrame(potential_list)
pot_df['date'] = pd.to_datetime(pot_df['date'].astype(str))

ax = pot_df.set_index('date')[['long_signal_pct', 'short_signal_pct']].plot()
ax.figure.savefig(f'label/{target_name}_imbalance.png')

short_wealth = pd.concat(wealth_factor['short'])
short_wealth.name = 'short_wealth'
long_wealth = pd.concat(wealth_factor['long'])
long_wealth.name = 'long_wealth'

pd.concat([long_wealth, short_wealth], axis=1).reindex(data.index).to_csv(fr'label/{target_name}.csv')
