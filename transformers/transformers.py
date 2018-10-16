import pandas as pd
from string import digits
from nltk.tokenize import word_tokenize
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin



class ProdDescCleaner(BaseEstimator, TransformerMixin):
    '''
    Pipeline transformer for the PROD_DESC_BY_VENDOR feature.
    '''

    def __init__(self,
                 col_names_to_strip=['MANUFACTURE_PART_NUMBER',
                                     'MFR_PART_NO_BY_VENDOR',
                                     'UNIT_OF_MEASURE',
                                     'UNIT_OF_MEASURE_BY_VENDOR']):
        self.col_names_to_strip = col_names_to_strip


    def fit(self, X, y=None):
        '''
        X is expected to be a pandas dataframe.
        '''
        def _sub_parts_measures(m):
            return '' if m.group() in parts_and_measures else m.group()

        prod_desc = X['PROD_DESC_BY_VENDOR'].str.lower()
        parts_and_measures = set()
        for i in self.col_names_to_strip:
            unique_elements = set(X[i])
            for j in unique_elements:
                parts_and_measures.add(j)
        # strip measurement and manufacturer part numbers
        prod_desc = prod_desc.apply(lambda x: re.sub(r'\w+',
                                                     _sub_parts_measures,
                                                     x))
        #replace commas, dashes and underscores with a space
        punct_to_space_re = re.compile('|'.join([',', '_', '-']))
        prod_desc = prod_desc.str.replace(punct_to_space_re,' ')
        #regex to match all punctuation except % and .
        punct_re = re.compile(r'[^\w\s](?<!%|\.)')
        prod_desc = prod_desc.str.replace(punct_re,'')
        #strip duplicate and leading/trailing whitespace
        prod_desc = prod_desc.apply(lambda x: re.sub(' +', ' ',x)).str.strip()
        prod_desc = prod_desc.apply(lambda x: x.lstrip(digits))
        self.X = prod_desc

        return self


    def transform(self, X, y=None):
        return self.X

    
    def execute(self,df):
        #df = pd.read_csv('H:/os3_taxonomy_constructor/data/os3_train.csv')
        #df = df.sample(n=20)
        self.fit(df)
        X = self.transform(df)
        return X
    
if __name__=='__main__':
    df = pd.read_csv('H:/os3_taxonomy_constructor/data/os3_train.csv')
    df = df.sample(n=20)
    pdc = ProdDescCleaner()
    pdc.fit(df)
    X = pdc.transform(df)
