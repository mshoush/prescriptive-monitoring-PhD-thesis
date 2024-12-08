from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from time import time

class IndexBasedTransformer(TransformerMixin):
    
    def __init__(self, results_dir, dataset_name, case_id_col, cat_cols, num_cols, max_events=None, model="catboost", fillna=True, create_dummies=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        
        self.model = model
        self.dataset_name = dataset_name
        self.results_dir = results_dir
        
        #self.boolean = True
        self.max_events = max_events
        self.fillna = fillna
        
        self.columns = None
        
        self.fit_time = 0
        self.transform_time = 0
        self.create_dummies = create_dummies
        
        
    

    
    
    def fit(self, X, y=None):
        return self
    

   
    def transform(self, X, y=None):
        #print(f"self.case_id_col:{self.case_id_col}")
        start = time()
        
        grouped = X.groupby(self.case_id_col, as_index=False)
        
        if self.max_events is None:
            self.max_events = grouped.size().max().values[1]
        
                   
        dt_transformed = pd.DataFrame(grouped.apply(lambda x: x.name), columns=[self.case_id_col])
       
        for i in range(int(self.max_events)):
            dt_index = grouped.nth(i)[[self.case_id_col] + self.cat_cols + self.num_cols]
            dt_index.columns = [self.case_id_col] + ["%s_%s"%(col, i) for col in self.cat_cols] + ["%s_%s"%(col, i) for col in self.num_cols]
            dt_transformed = pd.merge(dt_transformed, dt_index, on=self.case_id_col, how="left")
        dt_transformed.index = dt_transformed[self.case_id_col]
        
        # one-hot-encode cat cols
        if self.create_dummies and self.model!="catboost":
            all_cat_cols = ["%s_%s"%(col, i) for col in self.cat_cols for i in range(self.max_events)]
            dt_transformed = pd.get_dummies(dt_transformed, columns=all_cat_cols).drop(self.case_id_col, axis=1)
        elif self.model=="catboost":
            dt_transformed = dt_transformed.drop(self.case_id_col, axis=1)
            
            

        # fill missing values with 0-s
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)

        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]

        self.transform_time = time() - start
        
       #print("Save Index encoding")
        #print(dt_transformed)
        import os
        #dt_transformed.to_csv('dt_transformed_agg_%s.csv'%self.dataset_name, index=False, sep=';')
        #dt_transformed.to_csv(os.path.join(self.results_dir, 'dt_transformed_index_%s_%s.csv'%(self.model, self.dataset_name)),  index=False, sep=';')
        #dt_transformed.to_parquet(os.path.join(self.results_dir, 'dt_transformed_index_%s_%s.parquet'%(self.model, self.dataset_name)))
        cat_feat_idx = np.where((dt_transformed.dtypes == 'object'))[0]           
        #print(f"index: {cat_feat_idx}")
        if cat_feat_idx is not None:
            dt_transformed.iloc[:, cat_feat_idx] = dt_transformed.iloc[:, cat_feat_idx].astype(str)
        #print("Save index")
        dt_transformed.to_parquet(os.path.join(self.results_dir, 'dt_transformed_index_%s_%s.parquet'%(self.model, self.dataset_name)))
        #print("Saved index")
        #print("Saved Index encoding")
        import gc 
        gc.collect()
        
        return dt_transformed