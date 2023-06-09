import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import KFold

from typing import List


class CombinedSequence(Sequence):
    def __init__(self, *sequence_list, shuffle=False, seed=420, replace=False):
        self.sequence_list: List[Sequence] = sequence_list
        self.sub_sequence_len = tuple(len(s) for s in sequence_list)
        self.shuffle = shuffle
        self.replace = replace
        self.seed = seed

        if self.shuffle:
            self._random = np.random.RandomState(seed)
            self._map = self._randomize_map()
            
    def _randomize_map(self):
        return self._random.choice(np.arange(sum(self.sub_sequence_len)), 
                                   sum(self.sub_sequence_len),
                                   replace=self.replace)

    def __len__(self):
        return sum(self.sub_sequence_len)

    def on_epoch_end(self):
        for sq in self.sequence_list:
            sq.on_epoch_end()

        if self.shuffle:
            self._map = self._randomize_map()

    def __getitem__(self, input_idx):
        if self.shuffle:
            input_idx = self._map[input_idx]

        for sq, sub_l in zip(self.sequence_list, self.sub_sequence_len):
            if input_idx>=sub_l:
                input_idx -= sub_l
            else:
                break
        else:
            raise IndexError(f'Index {input_idx+len(self)} out of bound for Sequence of size {len(self)}')

        return sq[input_idx]


class StrideData(Sequence):
    def __init__(self, x_set, y_set, lookback, batch_size=1, batch_no=None, shuffle=False, replace=False, weights=None):
        if type(x_set) is pd.Series:
            self.x = x_set.to_frame().values
            self.x_index = x_set.index
            self.x_col = [x_set.name]
        elif type(x_set) is pd.DataFrame:
            self.x = x_set.values
            self.x_index = x_set.index
            self.x_col = x_set.columns
        else:
            self.x = x_set
            
        if type(y_set) is pd.Series:
            self.y = y_set.to_frame().values
            self.y_index = y_set.index
            self.y_col = [y_set.name]
        elif type(y_set) is pd.DataFrame:
            self.y = y_set.values
            self.y_index = y_set.index
            self.y_col = y_set.columns
        else:
            self.y = y_set
            
        self.lookback = lookback
        self.batch_size = batch_size
        
        self.default_length = int(np.floor(((len(self.x) + 1 - self.lookback) / self.batch_size)))
        self.batch_no = int(np.floor(((len(self.x) + 1 - self.lookback) / self.batch_size))) if batch_no is None else batch_no
        self.idx_range = np.arange(0, self.default_length * self.batch_size)
        
        self.shuffle = shuffle
        if replace:
            self.idx_range = np.random.choice(self.idx_range, 
                                              size=self.batch_no * self.batch_size, 
                                              replace=True)
            
        else:
            tile_count = np.ceil(self.batch_no * self.batch_size / self.idx_range.shape[0]).astype(int)
            size = self.batch_no * self.batch_size
            self.idx_range = np.hstack([self.idx_range for i in range(tile_count)])[:size]
            
            if shuffle:
                self.idx_range = np.random.choice(self.idx_range, 
                                                  size=size, 
                                                  replace=replace) # should be false
                
        self.weights = weights

    def __len__(self):
        return self.batch_no
    
    def _get_idx_range_list(self, idx_input):
        if idx_input < 0:
            idx_input = len(self) + idx_input
        idx = idx_input * self.batch_size
        return [np.arange(self.idx_range[idx + i], self.idx_range[idx + i] + self.lookback) for i in range(self.batch_size)]
    
    def get_indices(self, idx_input):
        idx_range_list = self._get_idx_range_list(idx_input)
        return [self.x_index[ir] for ir in idx_range_list]
    
    def get_target_indices(self, idx_input):
        idx_range_list = self._get_idx_range_list(idx_input)
        return [self.x_index[ir][-1] for ir in idx_range_list]

    def __getitem__(self, idx_input):
        idx_range_list = self._get_idx_range_list(idx_input)
        batch_x_list = [self.x[idx_range] for idx_range in idx_range_list]
        batch_y_list = [self.y[idx_range[-1]] for idx_range in idx_range_list]
        
        if self.weights is None:
            out = np.stack(batch_x_list), np.stack(batch_y_list)
        else:
            out = np.stack(batch_x_list), np.stack(batch_y_list), self.weights
        
        return out
    
    def on_epoch_end(self):
        if self.shuffle:
            self.idx_range = np.random.choice(self.idx_range, size=self.idx_range.shape)
    
    def get_df(self, idx_input):
        X, y = self[idx_input]
        X = [x for x in X]
        idx_list = self.get_indices(idx_input)
        x_col = self.x_col
        
        df_list = []
        for X_slice, idx in zip(X, idx_list):
            X_df = pd.DataFrame(X_slice, index=idx, columns=x_col)
            df_list.append(X_df)
            
        return df_list
    
    def get_target_df(self, idx_input):
        X, y = self[idx_input]
        y = [y_[-1] for y_ in y]
        idx = self.get_target_indices(idx_input)
        y_col = self.y_col
            
        return pd.DataFrame(y, index=idx, columns=y_col)
    

class SubSequence(Sequence):
    def __init__(self, sd: StrideData, selected_idx, batch_size=None, add_last_batch=False):
        self.batch_size = sd.batch_size if batch_size is None else batch_size
        self.sd = sd

        self.key_dict = {}
        query_dict = {}
        batch_count = 0
        sequence_count = 0

        for i in range(len(sd)):
            target_idx = sd.get_target_indices(i)
            selected_bool = pd.Index(target_idx).isin(selected_idx)
            batch_count += selected_bool.sum()

            if batch_count < self.batch_size:
                query_dict[i] = np.argwhere(selected_bool)[:,0]
            
            else:
                batch_count -= self.batch_size
                query_arr = np.argwhere(selected_bool)[:,0]
                cut = query_arr.shape[0] - batch_count

                query_dict[i] = query_arr[:cut]
                self.key_dict[sequence_count] = query_dict

                query_dict = {}
                query_dict[i] = query_arr[cut:]
                sequence_count += 1

        if add_last_batch:
            self.key_dict[sequence_count] = query_dict

    def get_query_dict(self, input_idx):
        if input_idx < 0:
            input_idx = len(self.key_dict) - input_idx
        return self.key_dict[input_idx]

    def __len__(self):
        return len(self.key_dict)

    def __getitem__(self, input_idx):
        query_dict = self.get_query_dict(input_idx)
        X_list = []
        y_list = []
        for k, query in query_dict.items():
            # print(query)
            X, y = self.sd[k]
            X_list.append(X[query])
            y_list.append(y[query])
            # print(y[query].mean(0))
        return np.vstack(X_list), np.vstack(y_list)

    def on_epoch_end(self):
        self.sd.on_epoch_end()

    def get_target_indices(self, input_idx):
        query_dict = self.get_query_dict(input_idx)
        indices_list = []
        for k, query in query_dict.items():
            idx = pd.Index(self.sd.get_target_indices(k))[query]
            indices_list.append(idx.values)

        return np.hstack(indices_list)
    
    
class SequencePair:
    def __repr__(self):
        sequence_str = str(self.train_sequence)
        index_str = '\nTrain indices are:\n'+str(self.train_index) if self.train_index is not None else ''
        index_str2 = '\nTest indices are:\n'+str(self.test_index) if self.test_index is not None else ''
        return sequence_str + index_str + index_str2
    
    def __init__(self, train_sequence : StrideData, X_test, y_test, train_index=None, test_index=None, weights=None):
        self.train_sequence: StrideData =train_sequence
        self.X_test=X_test
        self.y_test=y_test
        self.weights=weights
        self.test_index=test_index
        self.train_index=train_index
    
    @property
    def test_tuple(self):
        if self.weights is None:
            return (self.X_test, self.y_test)
        else:
            return (self.X_test, self.y_test, self.weights)
    
    
def create_train_val_sequence_cv(X, y, cv=4, lookback=252, 
                                  batch_size=25, batch_no=None, 
                                  shuffle=True, trim='default', 
                                  weights=False, focused_periods=252,
                                  replace=False):
    splitter = KFold(cv)
    
    trim = (0, lookback) if trim == 'default' else trim
    
    # X_array = X.values
    y_array = y.values
    y_count = y_array.shape[-1] if len(y_array.shape) == 2 else 1
    for train, test in [s for s in splitter.split(X, y)][::-1]:
        test_left = test[0] - trim[0]
        test_right = test[-1] + trim[1]
        train_trimmed = train[(train < test_left) | (train > test_right)]
        if weights:
            # TODO: label imbalance based weight
            # weights_arr = np.vstack([np.hstack([np.zeros(lookback-focused_periods), np.ones(focused_periods)])] * batch_size)
            # weights_arr = np.stack([weights_arr.reshape(batch_size, lookback)] * y_count, axis=-1)
            pass
        else:
            weights_arr = None
            
        train_s = StrideData(X.iloc[train_trimmed], y.iloc[train_trimmed], 
                             batch_size=batch_size, batch_no=batch_no, 
                             lookback=lookback, shuffle=shuffle, weights=weights_arr, 
                             replace=replace)

        left_test = max(test[0] - lookback - 1, 0)
        right_test = test[-1]
        max_test_batch = right_test - left_test - lookback
        if weights:
            weights_arr = np.vstack([np.hstack([np.zeros(lookback-focused_periods), np.ones(focused_periods)])] * max_test_batch)
            weights_arr = np.stack([weights_arr.reshape(max_test_batch, lookback)] * y_count, axis=-1)
        else:
            weights_arr = None
            
        test_s = StrideData(X.iloc[left_test:right_test], y.iloc[left_test:right_test], 
                            lookback=lookback, batch_size=max_test_batch, shuffle=False)
        X_test, y_test = test_s[0]
        if len(y_test.shape) > 2:
            y_test = y_test[:,-1,:]
            
        test_index = X.index[test]
        train_index = X.index[train_trimmed]

        sp = SequencePair(train_sequence=train_s, 
                          X_test=X_test, 
                          y_test=y_test,
                          weights=weights_arr,
                          train_index=train_index,
                          test_index=test_index)

        # sequence_pair_list.append(sp)
        yield sp
        
