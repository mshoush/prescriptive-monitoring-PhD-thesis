import sys

import dataset_confs

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


class DatasetManager:
    
    def __init__(self, dataset_name,  task_type="classification"):
        self.dataset_name = dataset_name
        self.task_type = task_type
        #print(self.dataset_name)
        self.case_id_col = dataset_confs.case_id_col[self.dataset_name]
        self.activity_col = dataset_confs.activity_col[self.dataset_name]
        try:
            self.resource_col = dataset_confs.resource_col[self.dataset_name]
        except:
            pass
        self.timestamp_col = dataset_confs.timestamp_col[self.dataset_name]
        if self.task_type=="classification":
            self.label_col = 'label'
        elif self.task_type=="regression":            
            self.label_col = 'remtime'
            
        #self.label_col = dataset_confs.label_col[self.dataset_name]
        self.pos_label = dataset_confs.pos_label[self.dataset_name]
        self.treatment_col = dataset_confs.treatment_col[self.dataset_name]
        self.pos_treatment = dataset_confs.pos_treatment[self.dataset_name]
        self.dynamic_cat_cols = dataset_confs.dynamic_cat_cols[self.dataset_name]
        self.static_cat_cols = dataset_confs.static_cat_cols[self.dataset_name]
        self.dynamic_num_cols = dataset_confs.dynamic_num_cols[self.dataset_name]
        self.static_num_cols = dataset_confs.static_num_cols[self.dataset_name]
        self.sorting_cols = [self.timestamp_col, self.activity_col]
      #  self.cat_feat = dataset_confs.cat_feat[self.dataset_name] # cat feat
        
    
    def read_dataset(self):
        # read dataset
        dtypes = {col:"object" for col in self.dynamic_cat_cols+self.static_cat_cols+[self.case_id_col, self.label_col, self.treatment_col, self.timestamp_col]}
        for col in self.dynamic_num_cols + self.static_num_cols:
            dtypes[col] = "float"

        #data = pd.read_csv(dataset_confs.filename[self.dataset_name], sep=";", dtype=dtypes)
        data = pd.read_parquet(dataset_confs.filename[self.dataset_name]).astype(dtypes)
        #print(data.head)
        #print(data.dtypes)
        data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col])

        return data

    # def read_dataset_pickle(self):
    #     with open(dataset_confs.filename[self.dataset_name], "rb") as fh:
    #         data = pickle.load(fh)
    #     # read dataset
    #     dtypes = {col:"object" for col in self.dynamic_cat_cols+self.static_cat_cols+[self.case_id_col, self.label_col, self.treatment_col, self.timestamp_col]}
    #     for col in self.dynamic_num_cols + self.static_num_cols:
    #         dtypes[col] = "float"
    #
    #     data = pd.read_csv(dataset_confs.filename[self.dataset_name], sep=";", dtype=dtypes)
    #     data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col])
    #
    #     return data


    # def split_data(self, data, train_ratio, split="temporal", seed=22):  
    #     # split into train and test using temporal split
    #     grouped = data.groupby(self.case_id_col)
    #     start_timestamps = grouped[self.timestamp_col].min().reset_index()
    #     if split == "temporal":
    #         start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
    #     elif split == "random":
    #         np.random.seed(seed)
    #         start_timestamps = start_timestamps.reindex(np.random.permutation(start_timestamps.index))

    #     train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
    #     train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
    #     test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')

    #     return (train, test)
    
    
    def split_data(self, data, train_ratio, val_ratio, split_type="temporal", seed=22):
        np.random.seed(seed)
        
        # Group data by case ID and find the start timestamp for each case
        grouped_data = data.groupby(self.case_id_col)
        start_timestamps = grouped_data[self.timestamp_col].min().reset_index()
        
        # Sort case IDs based on start timestamps
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind="mergesort")

        # Calculate sizes of train, val, and test sets
        total_cases = len(start_timestamps)
        train_size = int(train_ratio * total_cases)
        val_size = int(val_ratio * total_cases)
        test_size = total_cases - train_size - val_size

        # Divide case IDs into train, val, and test sets
        train_ids = list(start_timestamps[self.case_id_col])[:train_size]
        val_ids = list(start_timestamps[self.case_id_col])[train_size:train_size + val_size]
        test_ids = list(start_timestamps[self.case_id_col])[train_size + val_size:]

        # Filter data based on case IDs
        train = data[data[self.case_id_col].isin(train_ids)].reset_index(drop=True)
        val = data[data[self.case_id_col].isin(val_ids)].reset_index(drop=True)
        test = data[data[self.case_id_col].isin(test_ids)].reset_index(drop=True)

        return train, val, test

    
    
    
    
    def split_data_strict(self, data, train_ratio, split="temporal"):  
        # split into train and test using temporal split and discard events that overlap the periods
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        split_ts = test[self.timestamp_col].min()
        train = train[train[self.timestamp_col] < split_ts]
        return (train, test)
    
    def split_data_discard(self, data, train_ratio, split="temporal"):  
        # split into train and test using temporal split and discard events that overlap the periods
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        split_ts = test[self.timestamp_col].min()
        overlapping_cases = train[train[self.timestamp_col] >= split_ts][self.case_id_col].unique()
        train = train[~train[self.case_id_col].isin(overlapping_cases)]
        return (train, test)
    
    
    def split_val(self, data, val_ratio, split="random", seed=22):  
        # split into train and test using temporal split
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        elif split == "random":
            np.random.seed(seed)
            start_timestamps = start_timestamps.reindex(np.random.permutation(start_timestamps.index))
        val_ids = list(start_timestamps[self.case_id_col])[-int(val_ratio*len(start_timestamps)):]
        val = data[data[self.case_id_col].isin(val_ids)].sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        train = data[~data[self.case_id_col].isin(val_ids)].sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        return (train, val)
    
    def generate_prefix_data(self, data, min_length, max_length, gap=1):
        # generate prefix data (each possible prefix becomes a trace)
        data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

        dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length)
        dt_prefixes["prefix_nr"] = 1
        dt_prefixes["orig_case_id"] = dt_prefixes[self.case_id_col]
        #dt_prefixes[self.case_id_col] = dt_prefixes[self.case_id_col].astype('category')
        for nr_events in range(min_length+gap, max_length+1, gap):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
            tmp["orig_case_id"] = tmp[self.case_id_col]
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            #tmp[self.case_id_col] = tmp[self.case_id_col].astype('category')
            tmp["prefix_nr"] = nr_events
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        
        dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(max_length, x))
        #dt_prefixes[self.case_id_col] = dt_prefixes[self.case_id_col].astype('category')
        #dt_prefixes = dt_prefixes.reset_index(drop=True)
        
        
        return dt_prefixes
    
    

    # def generate_prefix_data(self, data, min_length, max_length, gap=1):
    #     # generate prefix data (each possible prefix becomes a trace)
    #     data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

    #     dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length)
    #     dt_prefixes["prefix_nr"] = 1
    #     dt_prefixes["orig_case_id"] = dt_prefixes[self.case_id_col]
    #     for nr_events in range(min_length+1, max_length+1):
    #         tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
    #         tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
    #         dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        
    #     dt_prefixes['case_length'] = dt_prefixes.groupby(self.case_id_col)[self.activity_col].transform(len)
        
    #     return dt_prefixes
        
        
        
        # for nr_events in range(min_length+gap, max_length+1, gap):
        #     tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
        #     tmp["orig_case_id"] = tmp[self.case_id_col]
        #     tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s"%(x, nr_events)).astype(str)
        #     tmp["prefix_nr"] = nr_events
        #     dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        
        # dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(max_length, x))
        
        # return dt_prefixes

    
    # def generate_prefix_data(self, data, min_length, max_length):
    #     # generate prefix data (each possible prefix becomes a trace)
    #     data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

    #     dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length)
    #     for nr_events in range(min_length+1, max_length+1):
    #         tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
    #         tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
    #         dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        
    #     dt_prefixes['case_length'] = dt_prefixes.groupby(self.case_id_col)[self.activity_col].transform(len)
        
    #     return dt_prefixes


    # def generate_prefix_data(self, data, min_length, max_length):
    #     # generate prefix data (each possible prefix becomes a trace)
    #     data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

    #     dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length)
    #     dt_prefixes["prefix_nr"] = 1
    #     dt_prefixes["orig_case_id"] = dt_prefixes[self.case_id_col]
    #     for nr_events in range(min_length+1, max_length+1):
    #         tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
    #         tmp["orig_case_id"] = tmp[self.case_id_col]
    #         tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
    #         tmp["prefix_nr"] = nr_events
    #         dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        
    #     dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(max_length, x))
        
    #     return dt_prefixes
    # def generate_prefix_data(self, data, min_length, max_length):
    #     # Generate prefix data (each possible prefix becomes a trace)
    #     data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

    #     dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length).copy(deep=True)
    #     dt_prefixes["prefix_nr"] = 1
    #     dt_prefixes["orig_case_id"] = dt_prefixes[self.case_id_col]
        
    #     for nr_events in range(min_length + 1, max_length + 1):
    #         tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events).copy()
    #         tmp["orig_case_id"] = tmp[self.case_id_col]
    #         tmp[self.case_id_col] = tmp[self.case_id_col].astype(str) + '_' + str(nr_events)
    #         tmp["prefix_nr"] = nr_events
    #         dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
            
    #     dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(max_length, x))
    #     print(dt_prefixes.dtypes)
        
    #     return dt_prefixes

    
    # def generate_prefix_data(self, data, min_length, max_length):
    #     # Generate prefix data (each possible prefix becomes a trace)
    #     data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

    #     dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length).copy(deep=True)
    #     dt_prefixes["prefix_nr"] = 1
    #     dt_prefixes["orig_case_id"] = dt_prefixes[self.case_id_col]
        
    #     for nr_events in range(min_length + 1, max_length + 1):
    #         tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events).copy()
    #         tmp["orig_case_id"] = tmp[self.case_id_col]
    #         tmp[self.case_id_col] = tmp[self.case_id_col] + '_' + str(nr_events)
    #         tmp["prefix_nr"] = nr_events
    #         dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
            
    #     dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(max_length, x))
        
    #     return dt_prefixes

    


    # def generate_prefix_data(self, data, min_length, max_length):
    #     # Generate prefix data (each possible prefix becomes a trace)
    #     data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

    #     # Filter data to retain only cases with sufficient length
    #     valid_cases = data[data['case_length'] >= min_length]

    #     # Initialize an empty DataFrame to store prefix data
    #     dt_prefixes = pd.DataFrame()

    #     # Generate prefixes for each possible length
    #     for nr_events in range(min_length, max_length + 1):
    #         # Filter data to retain only cases with at least nr_events
    #         tmp = data[data['case_length'] >= nr_events]
            
    #         # Generate case IDs with the current number of events
    #         #tmp[self.case_id_col] = tmp[self.case_id_col] + '_' + nr_events
    #         tmp[self.case_id_col] = tmp[self.case_id_col].astype(str) + '_' + str(nr_events)

            
    #         # Concatenate prefix data
    #         dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)

    #     # Recalculate case lengths after generating prefixes
    #     dt_prefixes['case_length'] = dt_prefixes.groupby(self.case_id_col)[self.activity_col].transform(len)

    #     return dt_prefixes



    def get_pos_case_length_quantile(self, data, quantile=0.90):
        if self.task_type=="classification":
            return int(np.ceil(data[data[self.label_col]==self.pos_label].groupby(self.case_id_col).size().quantile(quantile)))
        elif self.task_type=="regression":
            return int(np.floor(data.groupby(self.case_id_col).size().quantile(quantile)))
        
        

        
    
    

    def get_indexes(self, data):
        return data.groupby(self.case_id_col).first().index

    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]

    def get_label(self, data):
        return data.groupby(self.case_id_col).first()[self.label_col]

    def get_label_reg(self, data):
        return data.groupby(self.case_id_col).min()[self.label_col]

    #TODO get_treatment data
    def get_treatment(self, data):
        return data.groupby(self.case_id_col).first()[self.treatment_col]

    def get_case_ids(self, data, nr_events=1):
        case_ids = pd.Series(data.groupby(self.case_id_col).first().index)
        if nr_events > 1:
            case_ids = case_ids.apply(lambda x: "_".join(x.split("_")[:-1]))
        return case_ids
    
    def get_label_numeric(self, data):
        y = self.get_label(data)  # one row per case
        # neg_label[dataset] = "regular": 0
        # pos_label[dataset] = "deviant": 1
        return [1 if label == self.pos_label else 0 for label in y]
    
    def get_label_regression(self, data):
        y = self.get_label_reg(data)  # one row per case
        # neg_label[dataset] = "regular": 0
        # pos_label[dataset] = "deviant": 1
        return y#[1 if label == self.pos_label else 0 for label in y]

    #TODO adding treatment
    def get_treatment_numeric(self, data):
        t = self.get_treatment(data) # one row per case
        # neg_treatment[dataset] = "noTreat": 0
        # pos_treatment[dataset] = "treat": 1
        return [1 if treatment == self.pos_treatment else 0 for treatment in t]


    # TODO get timestamps
    def get_ts(self, data):
        #print(self.treatment_col)
        return data.groupby(self.case_id_col)[self.timestamp_col]

    # def get_ts(self, data):
    #     ts = self.get_ts(data)  # one row per case
    #     # neg_treatment[dataset] = "noTreat": 0
    #     # pos_treatment[dataset] = "treat": 1
    #     return ts

    def get_class_ratio(self, data):
        class_freqs = data[self.label_col].value_counts()
        return class_freqs[self.pos_label] / class_freqs.sum()
    
    # def get_stratified_split_generator(self, data, n_splits=5, shuffle=True, random_state=22):
    #     grouped_firsts = data.groupby(self.case_id_col, as_index=False).first()
    #     skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
    #     for train_index, test_index in skf.split(grouped_firsts, grouped_firsts[self.label_col]):
    #         current_train_names = grouped_firsts[self.case_id_col][train_index]
    #         train_chunk = data[data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
    #         test_chunk = data[~data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
    #         yield (train_chunk, test_chunk)
    
    # def get_stratified_split_generator_time_benshmarking(self, data, n_splits=5, shuffle=True, random_state=22):
    #     grouped_firsts = data.groupby(self.case_id_col, as_index=False).first()
    #     skf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
    #     for train_index, test_index in skf.split(grouped_firsts, grouped_firsts[self.label_col]):
    #         current_train_names = grouped_firsts[self.case_id_col][train_index]
    #         train_chunk = data[data[self.case_id_col].isin(current_train_names)]#.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
    #         test_chunk = data[~data[self.case_id_col].isin(current_train_names)]#.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
    #         yield (train_chunk, test_chunk)
            
    def get_idx_split_generator(self, dt_for_splitting, n_splits=5, shuffle=True, random_state=22):
        if self.label_col=="label":
            skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            data = dt_for_splitting
        elif self.label_col=="remtime":
            grouped_firsts = dt_for_splitting.groupby(self.case_id_col, as_index=False).first()
            skf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            data = grouped_firsts
        
         
        for train_index, test_index in skf.split(data, data[self.label_col]):
            current_train_names = data[self.case_id_col][train_index]
            current_test_names = data[self.case_id_col][test_index]
            yield (current_train_names, current_test_names)
            
    # def get_stratified_split_generator(self, data, n_splits=5, shuffle=True, random_state=22):
    #     grouped_firsts = data.groupby(self.case_id_col, as_index=False).first()
    #     skf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
    #     for train_index, test_index in skf.split(grouped_firsts, grouped_firsts[self.label_col]):
    #         current_train_names = grouped_firsts[self.case_id_col][train_index]
    #         train_chunk = data[data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
    #         test_chunk = data[~data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
    #         yield (train_chunk, test_chunk)
            
