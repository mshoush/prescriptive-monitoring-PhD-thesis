from sklearn.base import TransformerMixin
import pandas as pd
from time import time

class LastStateTransformer(TransformerMixin):
    
    def __init__(self, results_dir, dataset_name, case_id_col, cat_cols, num_cols, boolean=False, model="catboost", fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        
        self.model = model
        self.dataset_name = dataset_name
        self.results_dir = results_dir
        
        self.boolean = True
        self.fillna = fillna
        
        self.columns = None
        
        self.fit_time = 0
        self.transform_time = 0
    

    
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X, y=None):
        start = time()
        
        dt_last = X.groupby(self.case_id_col).last()
        
        # transform numeric cols
        dt_transformed = dt_last[self.num_cols]
        
        # # transform cat cols
        # # transform cat cols
        # if self.model!="catboost":
        #     dt_transformed = pd.get_dummies(X[self.cat_cols])
        #     dt_transformed[self.case_id_col] = X[self.case_id_col]
        #     del X
        # else:
        #     #dt_transformed = pd.get_dummies(X[self.cat_cols])
        #     dt_transformed = X[self.cat_cols]
        #     dt_transformed[self.case_id_col] = X[self.case_id_col]
        #     del X
        # if self.boolean:
        #     dt_transformed = dt_transformed.groupby(self.case_id_col).max()
        # else:
        #     dt_transformed = dt_transformed.groupby(self.case_id_col).sum()
        
        # transform cat cols
        if len(self.cat_cols) > 0 and self.model!="catboost":
            dt_cat = pd.get_dummies(dt_last[self.cat_cols])
            dt_transformed = pd.concat([dt_transformed, dt_cat], axis=1)
        elif len(self.cat_cols) > 0 and self.model=="catboost":
            dt_cat = dt_last[self.cat_cols]
            dt_transformed = pd.concat([dt_transformed, dt_cat], axis=1)
        
        # fill NA with 0 if requested
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)
            
        # add missing columns if necessary
        if self.columns is not None:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]
        else:
            self.columns = dt_transformed.columns
        
        self.transform_time = time() - start
        
        #print("Save LastState encoding")
        #print(dt_transformed)
        import os
        #dt_transformed.to_csv('dt_transformed_agg_%s.csv'%self.dataset_name, index=False, sep=';')
        # dt_transformed.to_csv(os.path.join(self.results_dir, 'dt_transformed_laststate_%s_%s.csv'%(self.model, self.dataset_name)),  index=False, sep=';')
        dt_transformed.to_parquet(os.path.join(self.results_dir, 'dt_transformed_laststate_%s_%s.parquet'%(self.model, self.dataset_name)))
        
        return dt_transformed
    
    
