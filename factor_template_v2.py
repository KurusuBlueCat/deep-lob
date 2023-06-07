import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

save_path = './factor/'

class Factor:
    def __init__(self, factor_name):
        self.factor_name = factor_name

    def get_data(self):
        self.data = pd.read_csv('data\data_night_shifted_au.csv.gz')


    def feature1(self):
        data = self.data.copy()
        data['ask_amount'] = (data['ask_5'] * data['ask_size_5'] + data['ask_4'] * data['ask_size_4'] + data['ask_3'] * data[
                'ask_size_3'] + data['ask_2'] * data['ask_size_2'] + data['ask_1'] * data['ask_size_1'])
        data['bid_amount'] = (
                    data['bid_5'] * data['bid_size_5'] + data['bid_4'] * data['bid_size_4'] + data['bid_3'] * data[
                'bid_size_3'] + data['bid_2'] * data['bid_size_2'] + data['bid_1'] * data['bid_size_1'])
        data['x'] = data['ask_amount'] - data['bid_amount']
        self.factor = data[['time','x']]

    def feature2(self):
        # Spread percentage
        data = self.data.copy()
        data['spread'] = (data['ask_1'] - data['bid_1'])
        data['mid_price'] = (data['ask_1'] + data['bid_1'])/2
        data['x'] = data['spread']/data['mid_price']
        self.factor = data[['time', 'x']]

    def feature3(self, n=200):
        # avg spread
        data = self.data.copy()
        data['spread'] = (data['ask_1'] - data['bid_1'])
        data['x'] = data['spread'].rolling(n).mean().fillna(0)
        self.factor = data[['time', 'x']]

    def feature4(self):
        # Order book depth
        data = self.data.copy()
        data['x']= data[['ask_size_5','ask_size_4','ask_size_3','ask_size_2','ask_size_1',
                             'bid_size_5','bid_size_4','bid_size_3','bid_size_2','bid_size_1']].sum(axis=1)
        self.factor = data[['time', 'x']]

    def feature5(self):
        # Best bid-ask ratio
        data = self.data.copy()
        data['x']= data['bid_size_1'] / data['ask_size_1']
        self.factor = data[['time', 'x']]


    def feature6(self):
        # avg turnover
        data = self.data.copy()
        data['bid_volume'] = data[['bid_size_5','bid_size_4','bid_size_3','bid_size_2','bid_size_1']].sum(axis=1)
        data['ask_volume'] = data[['ask_size_5','ask_size_4','ask_size_3','ask_size_2','ask_size_1']].sum(axis=1)
        data['x']= data['bid_volume'] / data['ask_volume']
        self.factor = data[['time', 'x']]

    def feature7(self):
        # order price skew
        data = self.data.copy()
        data['x']= data[["bid_5", "bid_4", "bid_3", "bid_2", "bid_1",
                         "ask_1", "ask_2", "ask_3", "ask_4", "ask_5"]].skew(axis=1)
        self.factor = data[['time', 'x']]

    def feature8(self):
        # order price kurt
        data = self.data.copy()
        data['x']= data[["bid_5", "bid_4", "bid_3", "bid_2", "bid_1",
                         "ask_1", "ask_2", "ask_3", "ask_4", "ask_5"]].kurt(axis=1)
        self.factor = data[['time', 'x']]

    def feature9(self):
        # rolling return
        data = self.data.copy()
        data['mid_price'] = (data['ask_1'] + data['bid_1'])/2
        data['x'] = data['mid_price'].diff(0) / data['mid_price']
        self.factor = data[['time', 'x']]

    def feature10(self,n=100):
        # rank best bid
        data = self.data.copy()
        data['x'] = ((data['bid_1'].rolling(n).rank())/n*2 - 1).fillna(0)
        self.factor = data[['time', 'x']]

    def feature11(self,n=100):
        # rank best ask
        data = self.data.copy()
        data['x'] = ((data['ask_1'].rolling(n).rank())/n*2 - 1).fillna(0)
        self.factor = data[['time', 'x']]

    def feature12(self,n=5):
        # price impact
        data = self.data.copy()
        ask, bid, ask_v, bid_v = 0, 0, 0, 0
        for i in range(1, n + 1):
            ask += data[f'ask_{i}'] * data[f'ask_size_{i}']
            bid += data[f'bid_{i}'] * data[f'bid_size_{i}']
            ask_v += data[f'ask_size_{i}']
            bid_v += data[f'bid_size_{i}']
        ask = ask / ask_v
        bid = bid / bid_v

        data['x'] = -(data['ask_1'] - ask)/data['ask_1'] - (data['bid_1'] - bid)/data['bid_1']
        self.factor = data[['time', 'x']]

    def feature13(self):
        # cofi
        data = self.data.copy()
        a = data['bid_size_1'] * np.where(data['bid_1'].diff() >= 0, 1, 0)
        b = data['bid_size_1'].shift() * np.where(data['bid_1'].diff() <= 0, 1, 0)
        c = data['ask_size_1'] * np.where(data['ask_1'].diff() <= 0, 1, 0)
        d = data['ask_size_1'].shift() * np.where(data['ask_1'].diff() >= 0, 1, 0)
        data['x'] =  (a - b - c + d).fillna(0)
        self.factor = data[['time', 'x']]

    def feature14(self, n=100):
        # price range
        data = self.data.copy()
        data['x'] = (data['ask_1'].rolling(n).max() / data['ask_1'].rolling(n).min() - 1).fillna(0)
        self.factor = data[['time', 'x']]

    def feature15(self, n=100):
        # quasi ask
        data = self.data.copy()
        data['x'] = data['ask_1'].diff(1).abs().rolling(n).sum().fillna(0)
        self.factor = data[['time', 'x']]

    def feature16(self, n=100):
        # quasi bid
        data = self.data.copy()
        data['x'] = data['bid_1'].diff(1).abs().rolling(n).sum().fillna(0)
        self.factor = data[['time', 'x']]

    def first_location_of_maximum(x):
        max_value = max(x)
        for loc in range(len(x)):
            if x[loc] == max_value:
                return loc + 1

    # def feature17(self, n=20):
    #     # ask price idxmax
    #     data = self.data.copy()
    #     data['x'] = data['ask_1'].rolling(n).apply(self.first_location_of_maximum, engine='numba', raw=True).fillna(0)
    #     self.factor = data[['time', 'x']]
    #
    # def feature18(self, n=20):
    #     # bid price idxmax
    #     data = self.data.copy()
    #     data['x'] = data['bid_1'].rolling(n).apply(self.first_location_of_maximum, engine='numba', raw=True).fillna(0)
    #     self.factor = data[['time', 'x']]

    def mean_second_derivative_centra(x):
        sum_value = 0
        for i in range(len(x) - 5):
            sum_value += (x[i + 5] - 2 * x[i + 3] + x[i]) / 2
        return sum_value / (2 * (len(x) - 5))

    # def feature19(self, n=20):
    #     # center deri two ask
    #     data = self.data.copy()
    #     data['x'] = data['ask_1'].rolling(n).apply(self.mean_second_derivative_centra, engine='numba', raw=True).fillna(0)
    #     self.factor = data[['time', 'x']]
    #
    # def feature20(self, n=20):
    #     # center deri two bid
    #     data = self.data.copy()
    #     data['x'] = data['bid_1'].rolling(n).apply(self.mean_second_derivative_centra, engine='numba', raw=True).fillna(0)
    #     self.factor = data[['time', 'x']]

        
    def feature21(self, n=20):
        #  slpoe spread/depth
        data = self.data.copy()
        data['x']= (data['ask_1'] - data['bid_1']) / (
                    data['ask_size_1'] + data['bid_size_1']) * 2
        self.factor = data[['time', 'x']]
        
        
    def feature22(self, n=20):
        # 初探市场微观结构：指令单薄与指令单流——资金交易策略之四 成交价的对数减去中间价的对数
        data = self.data.copy()
        data['x']  = np.log(data['vwap'] / data['volume']) - np.log((data['bid_1'] + data['ask_1']) / 2)
        self.factor = data[['time', 'x']]
        
        
    def feature23(self, n=20):
        # 计算加权的盘口挂单量
        data = self.data.copy()
        w = [1 - (i - 1) / 5 for i in range(1, 6)]
        w = np.array(w) / sum(w)
        data['x'] = data['bid_size_1'] * w[0] + data['bid_size_2'] * w[1] + data['bid_size_3'] * w[2] + data[
            'bid_size_4'] * w[3] + data['bid_size_5'] * w[4]
        wa = data['ask_size_1'] * w[0] + data['ask_size_2'] * w[1] + data['ask_size_3'] * w[2] + data[
            'ask_size_4'] * w[3] + data['ask_size_5'] * w[4]
        self.factor = data[['time','x']]

    def feature24(self, n=20):
        # 计算加权的盘口挂单量
        data = self.data.copy()
        w = [1 - (i - 1) / 5 for i in range(1, 6)]
        w = np.array(w) / sum(w)
        data['x'] = data['ask_size_1'] * w[0] + data['ask_size_2'] * w[1] + data['ask_size_3'] * w[2] + data[
            'ask_size_4'] * w[3] + data['ask_size_5'] * w[4]
        self.factor = data[['time','x']]

    def save_factor(self,i):
        # self.factor.dropna(inplace=True)
        self.factor.reindex(self.data.index)
        self.factor.to_csv(save_path+f'{self.factor_name}_{i}.csv',index=False)
        print(f'{self.factor_name}_{i}.csv'+' saved!')
        
    

if __name__ == '__main__':
    my_factor = Factor("factor")
    my_factor.get_data()
    for i in range(22,25):
        function_name = f'feature{i}'
        if hasattr(my_factor, function_name) and callable(getattr(my_factor, function_name)):
            function = getattr(my_factor, function_name)
            function()
            my_factor.save_factor(i)
        else:
            print("Function not found")