import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

save_path = './factor/'

class Factor:
    def __init__(self, factor_name):
        self.factor_name = factor_name

    def get_data(self):
        self.data = pd.read_csv('prepared_data.csv')


    def calculate(self):
        data = self.data
        data['ask_amount'] = (data['ask_5'] * data['ask_size_5'] + data['ask_4'] * data['ask_size_4'] + data['ask_3'] * data[
                'ask_size_3'] + data['ask_2'] * data['ask_size_2'] + data['ask_1'] * data['ask_size_1'])
        data['bid_amount'] = (
                    data['bid_5'] * data['bid_size_5'] + data['bid_4'] * data['bid_size_4'] + data['bid_3'] * data[
                'bid_size_3'] + data['bid_2'] * data['bid_size_2'] + data['bid_1'] * data['bid_size_1'])
        data['x'] = data['ask_amount'] - data['bid_amount']
        self.factor = data[['timestamp','x']]

    def save_factor(self,):
        self.factor.dropna(inplace=True)
        self.factor.to_csv(save_path+f'{self.factor_name}.csv')

if __name__ == '__main__':

    my_factor = Factor("factor_1")
    my_factor.get_data()
    my_factor.calculate()
    my_factor.save_factor()

