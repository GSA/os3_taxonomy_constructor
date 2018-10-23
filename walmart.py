import sys, os
if os.name=='nt':
    try:
        sys.path.append("H:/")
    except:
        print('cannont append path')

import re, requests, pandas as pd

from nltk.corpus import words
from config import wall_mart_api_key as key
from transformers.transformers import ProdDescCleaner
from nltk.tokenize import word_tokenize
from os3_taxonomy_constructor import rules
from statistics import mode
from data import data

categories_to_search_for = ('Office','Walmart for Business','Electronics')
categories_to_ignore = ('Essentials for Tax Professionals')


use_cols = ['AWARD_VEHICLE',
            'CONTRACT_NUMBER_AWARD_PIID', 'ORDER_NUMBER','VENDOR_NAME',
            'FUNDING_AGENCY', 'ORDER_DATE', 'QUANTITY_OF_ITEM_SOLD',
            'AWARD_PRICE_PER_UNIT', 'UNIT_OF_MEASURE',
            'UNIT_OF_MEASURE_BY_VENDOR', 'TOTAL_PRICE',
            'DESCRIPTION_OF_DELIVERABLES', 'PROD_DESC_BY_VENDOR',
            'MANUFACTURE_NAME', 'MANUFACTURE_PART_NUMBER', 'MFR_NAME_BY_VENDOR',
            'MFR_PART_NO_BY_VENDOR']




def main(sample=200):
    '''
    main function 
    
    1. gets data saved in folder
    2. calls api and maps predictions
    3. applies rules that are common mistakes the api makes and corrects them
    4. compares results to previously labeled data
    5. appends predictions to previously predicted data
    6. saves file
    '''
    train_df = get_training_data(sampleNum=sample)    
    description_category_map = query_api_and_make_predictions(train_df)
    train_df = map_and_clean_predictions(train_df,description_category_map)
    train_df = rules.applyRulesToDF(train_df)
    try:
        taxonomy =  pd.read_csv('data/walmart_taxonomy.csv')
        train_df = compare_to_labeled(taxonomy,train_df)
    except:
        print('please add taxonomy data to data folder to compare results')
    try:    
        previousPredictions = pd.read_csv('data/walmart_query_predictions.csv',encoding='latin1')
        train_df =train_df.append(previousPredictions)
    except:
        print('please add predictions to /data/') 
    print("saving data")
    train_df.to_csv(r'data/walmart_query_predictions.csv',index=False)



def get_taxonomy(key=key,name=""):
    '''
    Gets the category taxonomy used by walmart.com to categorize items.

    Parameters:
        key (str): your api key

    Returns:
        r_json (dict): a dict representing the json repsonse
    '''
    url = f"http://api.walmartlabs.com/v1/taxonomy?apiKey={key}"
    r = requests.get(url)
    r_json = r.json()
    return r_json


def search_walmart(key, query, category_id=None, facet=False, brand=None, price_range=None,num_items='10'):
    '''
    Query the Walmart Search API, returning results in json.

    Parameters:
        key (str) - required:  your API key
        query (str) - required: your query (e.g. ipod)
        category_id (str) - optional: The category id of the category for search within a category.
                                      This should match an id field from the Taxonomy API.
        facet (bool) - optional: Boolean flag to enable facets. Default value is False.
        brand (str) - optional: Brand name to filter search results. `facet` must be set to True.
        price_range (list) - optional: list of two ints for price range query, e.g. `[100,200]` to filter
                             search results for products within the $100-200 price range. `facet`
                             must be set to True.

    Returns:
        r_json (dict): a dict representing the json response of the request.
    '''

    url = f'http://api.walmartlabs.com/v1/search?apiKey={key}&query={query}&numItems={num_items}'
    if category_id:
        url = url + f'&categoryId={category_id}'
    if facet:
        url = url + '&facet=on'
    if brand:
        assert(facet), "Cannot include a brand in API call while facet kwarg is False"
        url = url + f'&facet.filter=brand:{brand}'
    if price_range:
        assert(facet), "Cannot include a brand in API call while facet kwarg is False"
        sorted_price_range = sorted(price_range)
        low = price_range[0]
        high = price_range[1]
        url = url + f'&facet.range=price:[{low} TO {high}]'
    r = requests.get(url)
    r_json = r.json()
    return r_json


                            

def predict_sub_category(vendor_description_clean, key, category_id = '1229749'):
    '''
    Query Walmart's Search API, returning most common product taxonomy of the
    search results.

    Parameters:
        vendor_description_clean (str): the product description from that should
                                        be the Search API Query
        key (str): the api key
        category_id (str): The category id of the category for search within a category.
                          This should match an id field from the Taxonomy API. By defaul
                          it refers to the Office category.

    Returns:
        prediction (str): the most common product taxonomy of the search results.
    '''  

    r_json = search_walmart(key,vendor_description_clean , category_id = category_id)
    category_paths = []
    # this means results weren't found
    if 'message' in r_json:
        new_query = strip_misspellings(vendor_description_clean)
        if len(new_query) > 0:
            r_json = search_walmart(key, new_query, category_id = category_id)
            category_paths = loop_through_data_elements_and_append_categories(r_json)   
    else:
        category_paths = loop_through_data_elements_and_append_categories(r_json)
    prediction = _get_prediction_from_category_path(category_paths)
    return prediction



def loop_through_data_elements_and_append_categories(r_json):
    category_paths = []
    dict_elements = dict_generator(r_json)
    for i in dict_elements:
        for j in i:
            if 'categoryPath' in str(j):
                if (i[-1].startswith((categories_to_search_for)) and not i[-1].endswith((categories_to_ignore))):
                        category_paths.append(i[-1])
    return category_paths

def strip_nonsense(text):
    '''
    lowercase a string, strip digits and substrings with digits in them.

    Parameters:
        text (str)
    Returns:
        stripped_text (str)
    '''

    text_lowered = text.lower()
    nums_replaced = re.sub("\d+", " ", text_lowered)
    no_nonsense = re.findall(r'\b[a-z][a-z][a-z]+\b',nums_replaced)
    stripped_text = ' '.join(w for w in no_nonsense).strip()
    return stripped_text


def strip_misspellings(text):
        '''
        Strip misspelled substrings from a string.
        '''
        tokens = word_tokenize(text)
        
        
        non_dict_words = set([word for word in tokens if word not in words.words() and re.match('^[a-zA-Z ]*$',word)])
        stripped_text = " ".join([x for x in tokens if x not in non_dict_words])
        
        return stripped_text

def dict_generator(indict, pre=None):
        '''
        Flatten irregularly nested dictionaries into lists for easier looping.
        '''
        pre = pre[:] if pre else []
        if isinstance(indict, dict):
            for key, value in indict.items():
                if isinstance(value, dict):
                    for d in dict_generator(value, pre + [key]):
                        yield d
                elif isinstance(value, list) or isinstance(value, tuple):
                    for v in value:
                        for d in dict_generator(v, pre + [key]):
                            yield d
                else:
                    yield pre + [key, value]
        else:
            yield indict

def _get_prediction_from_category_path(category_paths):
    prediction = 'NULL'
    try:
        while category_paths:
            try:
                prediction = mode(category_paths)  
                break
            except:
                del category_paths[-1]
    except:
          print("no prediction generated")
    return prediction

def remove_previous_predictions(train_df):
    '''
    remove predictions made from previous uses of the api
    '''
    previousPredictions = pd.read_csv('data/walmart_query_predictions.csv',encoding='latin1')
    prod_desc = list(previousPredictions['PROD_DESC_BY_VENDOR'])
    df = train_df[~train_df['PROD_DESC_BY_VENDOR'].isin(prod_desc)]
    return df




def get_training_data(sampleNum=5):
    '''read in the training data and take a random sample of it that respects
    the sub_category balance.
    '''
    train_df = data().get_data_not_yet_predicted()
    print(len(train_df.index),' rows still left to predict')
    try:
        train_df = remove_previous_predictions(train_df)
    except:
        print("add previous predictions to your data/ folder, to not resample data")
    lables, uniques = pd.factorize(train_df['SUB_CATEGORY'])
    train_df['target'] = lables
    #factor_map = {k:v for k,v in zip(uniques,range(len(uniques)))}
    train_df = train_df.sample(n=sampleNum, weights='target')
    train_df['vendor_description_clean'] = train_df['PROD_DESC_BY_VENDOR'].apply(strip_nonsense)
    return train_df


 
def query_api_and_make_predictions(train_df):
    '''
    Query Walmart Search API using cleaned up vendor descriptions
    '''
    vendor_descriptions = set(train_df['vendor_description_clean'])
    cleaned_descriptions =ProdDescCleaner().execute(train_df)
    description_category_map = {k:None for k in cleaned_descriptions}
    for i, query in enumerate(vendor_descriptions):
        description_category_map[query] = predict_sub_category(query,key)
        if i % 50 == 0:
            print(f'Done with {i} of {len(vendor_descriptions)}!')
    return description_category_map


def getTaxonomyDF(category='office'):
    from pandas.io.json import json_normalize
    '''
    will build this out better
    gets walmarts taxonomy into a df structure
    '''
    taxonomy = get_taxonomy()
    if category == 'office':
        num = 19
    else:
        num = 31
    df = pd.DataFrame()
    for i in range(len(taxonomy['categories'][num]['children'])):
        name = taxonomy['categories'][num]['children'][i]['name']
        try:
            df2 = json_normalize(taxonomy['categories'][num]['children'][i]['children'])
            df2['sub_category']=name 
        except:
            df2 = json_normalize(taxonomy['categories'][num]['children'][i])
        df = df.append(df2)
    del df['id']
    del df['path']
    df['category'] =category
    return df   


def compare_to_labeled(taxonomy,predictions):
    '''
    compares predictions made in recent query to previously labeled data
    '''
  
    taxonomy = taxonomy.drop_duplicates(subset=['walmart_name'])
    merged = pd.merge(predictions,taxonomy,how='left',left_on='prediction_truncated',right_on='walmart_name')
    merged['ifTrue'] = merged.apply(lambda x : 1 if x['SUB_CATEGORY'] == x['os3_taxonomy'] else 0, axis=1)
    df = pd.DataFrame(merged.groupby(['ifTrue'])['ifTrue'].count())
    print(df['ifTrue'].iloc[1] / (df['ifTrue'].iloc[1] + df['ifTrue'].iloc[0])," percent similar to labeled data")
    return merged

def map_and_clean_predictions(train_df,description_category_map):
    '''
    maps predictions onto training data and cleans them
    '''
    train_df['prediction'] = train_df['vendor_description_clean'].map(description_category_map)
    train_df['prediction_truncated'] = train_df['prediction'].str.split("/").apply(lambda x: x[-1])
    train_df = train_df[['PROD_DESC_BY_VENDOR',
              'vendor_description_clean',
              'SUB_CATEGORY',
              'prediction',
              'prediction_truncated']]
    return train_df



if __name__ == '__main__':
    main()
    
    


