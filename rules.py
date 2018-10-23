# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:36:28 2018

@author: AustinPeel
"""
import numpy as np

'''
these are rules that go in place after walmart makes the prediction 

if both words appear in the description then we change the prediction_truncated

code can be changed to allow three words etc.. but I have not written that

'''
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
'rule12':{'Ink and Toner' : ['print', 'oem']},
'rule13':{'Notebooks':['notebook', 'college']},
'rule14':{'Notebooks':['business', 'notebook']},
'rule15':{'Ink and Toner':['cartridge', 'oem']},
'rule16':{'Tape':['CARTRIDGE', 'APE']},
'rule17':{'Notebooks':['composition', 'book']},
'rule18':{'Paper and Stationery':['laser', 'paper']},
'rule20':{'Writing Utensils':['marker', 'paint']},
'rule21':{'Ink and Toner':['oem', 'comp']},
'rule22':{'Ink and Toner':['oem', 'inkjet']},
'rule23':{'Paper and Stationery':['paper', 'rec']},
'rule25':{'Clips':['ppr', 'clips']},
'rule26':{'Notebooks':['rule', 'notebook']},
'rule27':{'Notebooks':['subject', 'notebook']},
'rule28':{'Tape':['tape', 'invisible']},
'rule29':{'Tape':['tape', 'laminated']},
'rule30':{'Tape':['tape', 'packing']},
'rule31':{'Tape':['tape', 'pckng']},
'rule32':{'Tape':['tape', 'pkg']},
'rule33':{'Tape':['tape', 'scotch']},
'rule34':{'Tape':['tapedispnsr', 'tape']}

}

def _at_least_n_strings_in_longstring(longstring,strings,n=2): 
    strings = iter(strings)
    return all(any(string in longstring for string in strings) for _ in range(n))

def applyRulesToDF(df):
    print('applying rules')
    for i in ruleUpdates:
        for rule in ruleUpdates[i]:
            df['new'] = df.apply(lambda x : _at_least_n_strings_in_longstring(x['vendor_description_clean'],strings=[ruleUpdates[i][rule][0],ruleUpdates[i][rule][1]],n=2), axis=1)
            df['prediction_truncated'] = np.where(df['new']==True, rule, df['prediction_truncated'])
    del df['new']
    return df

