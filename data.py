# -*- coding: utf-8 -*-
"""
this class will help with data manaement and all import and export should be controlled here

this should be broken into two class methods [ get and send ]  when we have database up and ready


"""

import pandas as pd, numpy as np

dtypes = {'AWARD_VEHICLE': 'object',
          'CONTRACT_NUMBER_AWARD_PIID': 'object',
          'ORDER_NUMBER': 'object',
          'VENDOR_NAME': 'object',
          'FUNDING_AGENCY': 'object',
          'QUANTITY_OF_ITEM_SOLD': 'int',
          'AWARD_PRICE_PER_UNIT': 'float64',
          'UNIT_OF_MEASURE': 'object',
          'UNIT_OF_MEASURE_BY_VENDOR': 'object',
          'TOTAL_PRICE': 'float64',
          'DESCRIPTION_OF_DELIVERABLES': 'object',
          'PROD_DESC_BY_VENDOR': 'object',
          'MANUFACTURE_NAME': 'object',
          'MANUFACTURE_PART_NUMBER': 'object',
          'MFR_NAME_BY_VENDOR': 'object',
          'MFR_PART_NO_BY_VENDOR': 'object'}


class data():
    
    def __init__(self):
        previousPredictions = pd.read_csv('data/walmart_query_predictions.csv',encoding='latin1')
        self.train_df = pd.read_csv('data/os3_train.csv',dtype=dtypes,thousands=',',parse_dates=[7],encoding='latin1')        
        self.p = previousPredictions.drop_duplicates(subset=['PROD_DESC_BY_VENDOR'])
        
        
    def get_data_already_predicted(self,change_predictions=True):
        if change_predictions:
            self.p = self._change_predictions()
        merged = pd.merge(self.train_df,self.p,how='inner',on='PROD_DESC_BY_VENDOR')
        return merged

    def get_data_not_yet_predicted(self):
        new = self.train_df[~self.train_df['PROD_DESC_BY_VENDOR'].isin(self.p['PROD_DESC_BY_VENDOR'])].dropna()
        new = new.drop_duplicates(subset=['PROD_DESC_BY_VENDOR'])
        return new

    def get_training(self):
        return self.train_df
    
    def _change_predictions(self):
        self.p['prediction_truncated'] = np.where(self.p['is_walmart_right']=='1', self.p['prediction_truncated'],self.p['SUB_CATEGORY'])
        return self.p










