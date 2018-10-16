# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 09:27:46 2018

@author: AustinPeel
"""

import pandas as pd


taxonomy = pd.read_csv('H:/os3_taxonomy_constructor/data/walmart_taxonomy.csv')
taxonomy = taxonomy.drop_duplicates(subset=['name'])
df = pd.read_csv('H:/os3_taxonomy_constructor/data/walmart_query_predictions.csv')
merged = pd.merge(df,taxonomy,how='left',left_on='prediction_truncated',right_on='name')

merged['ifTrue'] = merged.apply(lambda x : 1 if x['SUB_CATEGORY'] == x['os3_taxonomy'] else 0, axis=1)

df = pd.DataFrame(merged.groupby(['ifTrue'])['ifTrue'].count())
df['ifTrue'].iloc[1] / (df['ifTrue'].iloc[1] + df['ifTrue'].iloc[0])
a = merged[merged['ifTrue']==0]
