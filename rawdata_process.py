import pandas
import os

import pandas as pd

data_list = []
folder_path = "."  # 当前文件夹路径
files = os.listdir(folder_path)  # 获取文件夹中所有文件和文件夹的列表

csv_gz_files = [file for file in files if file.startswith("SHFE")]
for file in csv_gz_files:
    data = pd.read_csv(f"E:\csi_level2\{file}\{file}/bu2309.csv",encoding="gbk")
    data_list.append(data)
data_total = pd.concat(data_list)
data_final = data_total[['交易日','最新价',
                         '上次结算价','昨收盘',
                         '昨持仓量','今开盘',
                         '最高价','最低价',
                         '数量','成交金额',
                         '持仓量','最后修改时间',
                         '最后修改毫秒',
                         '申买价一','申买量一',
                         '申卖价一','申卖量一',
                         '申买价二','申买量二',
                         '申卖价二','申卖量二',
                         '申买价三', '申买量三',
                         '申卖价三', '申卖量三',
                         '申买价四', '申买量四',
                         '申卖价四', '申卖量四',
                         '申买价五', '申买量五',
                         '申卖价五', '申卖量五',
                         '当日均价'
                         ]].rename(columns={'交易日':'date','最新价':'last_price',
                         '上次结算价':'last_settlement_price','昨收盘':'last_close',
                         '昨持仓量':'last_open_interest','今开盘':'open',
                         '最高价':'high','最低价':'low',
                         '数量':'volume','成交金额':'amount',
                         '持仓量':'open_interest','最后修改时间':'min',
                         '最后修改毫秒':'ms',
                         '申买价一':'bid_1','申买量一':'bid_size_1',
                         '申卖价一':'ask_1','申卖量一':'ask_size_1',
                         '申买价二':'bid_2','申买量二':'bid_size_2',
                         '申卖价二':'ask_2','申卖量二':'ask_size_2',
                         '申买价三':'bid_3', '申买量三':'bid_size_3',
                         '申卖价三':'ask_3', '申卖量三':'ask_size_3',
                         '申买价四':'bid_4', '申买量四':'bid_size_4',
                         '申卖价四':'ask_4', '申卖量四':'ask_size_4',
                         '申买价五':'bid_5', '申买量五':'bid_size_5',
                         '申卖价五':'ask_5', '申卖量五':'ask_size_5',
                         '当日均价':'vwap'})

data_final['timestamp'] = data_final['date'].astype(str)+' '+data_final['min'].astype(str)+data_final['ms'].astype(str)
data_final.to_csv('prepared_data.csv',index=False)