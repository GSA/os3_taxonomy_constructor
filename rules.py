# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:36:28 2018

@author: AustinPeel
"""
import numpy as np

ruleUpdates = {
'rule1':{"Notebooks":["steno" ,"book"]},
'rule2':{"Tape" :["tape","pkg"]},
'rule3':{'Writing Utensils' : ['marker','perm']},
'rule4':{'Ink and Toner' : ['toner', 'cartridge']},
'rule5':{'Labels' :['laser','label']},
'rule6':{'Labels' :['label' ,'address']},
'rule7':{'Labels' :['label' ,'lsr']},
'rule8':{'Clips, Clamps and Rings' :['Binder' ,'Clips']},
'rule9':{'Labels' :['label' ,'ship']},
'rule10':{'Ink and Toner' : ['toner', 'oem']},
'rule11':{'Ink and Toner' : ['print', 'cartridge']},
'rule12':{'Ink and Toner' : ['print', 'oem']}
}

def _at_least_n_strings_in_longstring(longstring,strings,n=2): 
    strings = iter(strings)
    return all(any(string in longstring for string in strings) for _ in range(n))

def applyRulesToDF(df):
    for i in ruleUpdates:
        for rule in ruleUpdates[i]:
            print('applying ', rule)
            df['new'] = df.apply(lambda x : _at_least_n_strings_in_longstring(x['vendor_description_clean'],strings=[ruleUpdates[i][rule][0],ruleUpdates[i][rule][1]],n=2), axis=1)
            df['prediction_truncated'] = np.where(df['new']==True, rule, df['prediction_truncated'])
    del df['new']
    return df

