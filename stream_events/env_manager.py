import pandas as pd
import os
import sys

# Define column names
case_id_col = "case_id"
timestamp_col = "timestamp"

class EnvManager:
    def __init__(self,data_type, dataset_name, results_dir, file_name, data_dir):
        # Initialize attributes
        self.dataset_name = dataset_name
        self.results_dir = results_dir
        self.current_case_id = 0
        self.finished = False
        self.nr_cases = 0
        self.treated_cases = []
        self.index = 0
        self.done = 0
        self.open_cases = {}
        self.finished_cases = {}       
        
        
        # Define directory and filename
        self.directory = results_dir
        self.filename = file_name
        self.data_dir = data_dir
        
        self.case_id_col = "case_id"
        self.activity_col = "activity"
        self.resource_col = "resource"
        self.timestamp_col = "timestamp"
        self.label_col = "label"
        self.treatment_col = "Treatment1"
        
        # Make results directory if not exists
        self.make_results_dir()
        
        # Load cases and sort events
        self.load_cases()
        self.sort_events()
        self.get_num_of_cases()

    def make_results_dir(self):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)  
    def return_df(self):
        # Read data from parquet file
        self.df = pd.read_parquet(os.path.join(self.data_dir, self.filename))
        self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
        self.df.sort_values(by=timestamp_col, inplace=True)
        return self.df
    
    def reset(self):
        self.current_case_id = 0
        self.finished = False
        self.nr_cases = 0
        self.treated_cases = []
        self.index = 0
        self.done = 0
        self.open_cases = {}
        self.finished_cases = {}
        

    def load_cases(self):
        # Read data from parquet file
        self.df = pd.read_parquet(os.path.join(self.data_dir, self.filename))
        self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
        self.df.sort_values(by=timestamp_col, inplace=True)

    def sort_events(self):
        # Sort events by timestamp
        self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
        self.df.sort_values(by=timestamp_col, inplace=True)

    def get_num_of_cases(self):
        # Get number of unique cases
        self.num_cases = self.df[case_id_col].nunique()
        return self.num_cases

    def get_event_by_event(self):
        if self.index != self.df.shape[0]:            
            # Get current event
            self.current_event = self.df.iloc[[self.index]]
            self.index += 1
            self.case_id = self.current_event[case_id_col].iloc[0]
            #print(f"Case started: {self.case_id}")
            if self.current_event["prefix_nr"].iloc[0] == self.current_event["case_length"].iloc[0]:
                # Case finished
                self.finished_cases[self.case_id] = [
                    self.current_event["prefix_nr"].iloc[0],
                    self.current_event["case_length"].iloc[0],
                ]                
                self.done = 1
                #print(f"Case ended: {self.case_id}")
            else:
                # Case ongoing
                self.open_cases[self.case_id] = [
                    self.current_event["prefix_nr"].iloc[0],
                    self.current_event["case_length"].iloc[0],
                ]
                self.done = 0
        else:
            self.done = 1
            self.finished = True
        return self.current_event
    