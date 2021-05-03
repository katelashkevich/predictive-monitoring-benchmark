import sys

import dataset_confs

import pandas as pd
import numpy as np
from nltk import ngrams

from sklearn.model_selection import StratifiedKFold


class DatasetManager:
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
        self.case_id_col = dataset_confs.case_id_col[self.dataset_name]
        self.activity_col = dataset_confs.activity_col[self.dataset_name]
        self.timestamp_col = dataset_confs.timestamp_col[self.dataset_name]
        self.label_col = dataset_confs.label_col[self.dataset_name]
        self.pos_label = dataset_confs.pos_label[self.dataset_name]

        self.dynamic_cat_cols = dataset_confs.dynamic_cat_cols[self.dataset_name]
        self.static_cat_cols = dataset_confs.static_cat_cols[self.dataset_name]
        self.dynamic_num_cols = dataset_confs.dynamic_num_cols[self.dataset_name]
        self.static_num_cols = dataset_confs.static_num_cols[self.dataset_name]
        
        self.sorting_cols = [self.timestamp_col, self.activity_col]
    
    def read_dataset(self):
        # read dataset
        dtypes = {col:"object" for col in self.dynamic_cat_cols+self.static_cat_cols+[self.case_id_col, self.label_col, self.timestamp_col]}
        for col in self.dynamic_num_cols + self.static_num_cols:
            dtypes[col] = "float"

        data = pd.read_csv(dataset_confs.filename[self.dataset_name], sep=";", dtype=dtypes)
        data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col])

        return data


    def split_data(self, data, train_ratio, split="temporal", seed=22):  
        # split into train and test using temporal split

        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        elif split == "random":
            np.random.seed(seed)
            start_timestamps = start_timestamps.reindex(np.random.permutation(start_timestamps.index))
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')

        return (train, test)
    
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
        for nr_events in range(min_length+gap, max_length+1, gap):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
            tmp["orig_case_id"] = tmp[self.case_id_col]
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            tmp["prefix_nr"] = nr_events
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        
        dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(max_length, x))
        
        return dt_prefixes

    # (NEW) N-Grams function
    def generate_ngram_data(self, data, n_size = 4, gap = 1):
        orig_case_id_col = 'orig_case_id'
        event_nr_col = 'event_nr'
        shift_col = 'ngram_shift'

        # Create empty result dataframe with same columns as input
        result = pd.DataFrame().reindex(columns=data.columns)
        result[event_nr_col] = None
        result[orig_case_id_col] = None
        result[shift_col] = None

        # Iterate over all cases
        for case_id in data[self.case_id_col].unique():
            # Get all events for the current case_id sorted by timestamp
            case_data = data[data[self.case_id_col] == case_id].sort_values([self.timestamp_col], ascending=True, kind='mergesort')
            case_data.insert(len(case_data.columns), event_nr_col, range(1, len(case_data) + 1))

            # Getting the original row indexes to use in N-gram copying
            event_indexes = case_data.index.values
            # If the provided n_size is bigger than the case lengtn, set N as case length
            N = min(n_size, len(event_indexes))

            # Get all N-grams for events in the current case
            ngrams_indexes = list(ngrams(event_indexes, N))

            for i in range(0, len(ngrams_indexes), gap):
                # Get the subset relevant to current N-gram in the original dataset
                ngram_subset_df = case_data.loc[list(ngrams_indexes[i])].copy()
                # Assign the original case id (just in case)
                ngram_subset_df[orig_case_id_col] = ngram_subset_df[self.case_id_col]
                ngram_subset_df[shift_col] = i
                # Modify case ID column to remain unique (concatenating N-gram number to the end)
                ngram_subset_df[self.case_id_col] = ngram_subset_df[self.case_id_col].apply(lambda x: "%s_%s"%(x, i))
                # Concatenate the N-gram subset to the result
                result = pd.concat([result, ngram_subset_df], axis=0)

        return result

    # def get_pos_case_length_quantile(self, data, quantile=0.90):
    #     return int(np.ceil(data[data[self.label_col]==self.pos_label].groupby(self.case_id_col).size().quantile(quantile)))
    
    def get_pos_case_length_quantile(self, data, quantile=0.90):
        return int(np.ceil(data.groupby(self.case_id_col).size().quantile(quantile)))

    def get_indexes(self, data):
        return data.groupby(self.case_id_col).first().index

    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]

    def get_label(self, data):
        return data.groupby(self.case_id_col).first()[self.label_col]
    
    def get_prefix_lengths(self, data):
        return data.groupby(self.case_id_col).last()["prefix_nr"]

    def get_ngram_shifts(self, data):
        return data.groupby(self.case_id_col).last()["ngram_shift"]
    
    def get_case_ids(self, data, nr_events=1):
        case_ids = pd.Series(data.groupby(self.case_id_col).first().index)
        if nr_events > 1:
            case_ids = case_ids.apply(lambda x: "_".join(x.split("_")[:-1]))
        return case_ids
    
    def get_label_numeric(self, data):
        y = self.get_label(data) # one row per case
        return [1 if label == self.pos_label else 0 for label in y]
    
    def get_class_ratio(self, data):
        class_freqs = data[self.label_col].value_counts()
        return class_freqs[self.pos_label] / class_freqs.sum()
    
    def get_stratified_split_generator(self, data, n_splits=5, shuffle=True, random_state=22):
        grouped_firsts = data.groupby(self.case_id_col, as_index=False).first()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        for train_index, test_index in skf.split(grouped_firsts, grouped_firsts[self.label_col]):
            current_train_names = grouped_firsts[self.case_id_col][train_index]
            train_chunk = data[data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            test_chunk = data[~data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            yield (train_chunk, test_chunk)
            
    def get_idx_split_generator(self, dt_for_splitting, n_splits=5, shuffle=True, random_state=22):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        for train_index, test_index in skf.split(dt_for_splitting, dt_for_splitting[self.label_col]):
            current_train_names = dt_for_splitting[self.case_id_col][train_index]
            current_test_names = dt_for_splitting[self.case_id_col][test_index]
            yield (current_train_names, current_test_names)
            