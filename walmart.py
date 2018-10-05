import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import requests
from bs4 import BeautifulSoup
from time import sleep
import gensim
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
import enchant


def get_taxonomy(key):
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


def search_walmart(key, query, category_id=None, facet=False, brand=None, price_range=None):
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

    url = f'http://api.walmartlabs.com/v1/search?apiKey={key}&query={query}'
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

    def strip_misspellings(text):
        '''
        Strip misspelled substrings from a string.
        '''
        d = enchant.Dict("en_US")
        tokens = word_tokenize(text)
        non_dict_words = set([word for word in tokens if d.check(word) is False and re.match('^[a-zA-Z ]*$',word)])
        stripped_text = " ".join([x for x in tokens if x not in non_dict_words])
        return stripped_text

    r_json = search_walmart(key, vendor_description_clean, category_id = category_id)
    # this means results weren't found
    if 'message' in r_json:
        new_query = strip_misspellings(vendor_description_clean)
        if len(new_query) > 0:
            r_json = search_walmart(key, new_query, category_id = category_id)
            dict_elements = dict_generator(r_json)
            category_paths = []
            for i in dict_elements:
                for j in i:
                    if 'categoryPath' in str(j):
                        category_paths.append(i[-1])
            try:
                prediction = max(category_paths)
            except ValueError:
                prediction = 'NULL'

            return prediction
        else:
            return 'NULL'
    else:
        dict_elements = dict_generator(r_json)
        category_paths = []
        for i in dict_elements:
            for j in i:
                if 'categoryPath' in str(j):
                    category_paths.append(i[-1])
        try:
            prediction = max(category_paths)
        except ValueError:
            prediction = 'NULL'

        return prediction

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



if __name__ == '__main__':
    # insert your api key here
    key = ''
    use_cols = ['AWARD_VEHICLE',
                'CONTRACT_NUMBER_AWARD_PIID', 'ORDER_NUMBER','VENDOR_NAME',
                'FUNDING_AGENCY', 'ORDER_DATE', 'QUANTITY_OF_ITEM_SOLD',
                'AWARD_PRICE_PER_UNIT', 'UNIT_OF_MEASURE',
                'UNIT_OF_MEASURE_BY_VENDOR', 'TOTAL_PRICE',
                'DESCRIPTION_OF_DELIVERABLES', 'PROD_DESC_BY_VENDOR',
                'MANUFACTURE_NAME', 'MANUFACTURE_PART_NUMBER', 'MFR_NAME_BY_VENDOR',
                'MFR_PART_NO_BY_VENDOR']
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
    # read in the training data and take a random sample of it that respects
    # the sub_category balance.
    train_df = pd.read_csv('os3_train.csv',
                           dtype=dtypes,
                           thousands=',',
                           parse_dates=[7],
                           encoding='latin1')
    lables, uniques = pd.factorize(train_df['SUB_CATEGORY'])
    train_df['target'] = lables
    factor_map = {k:v for k,v in zip(uniques,range(len(uniques)))}
    train_df = train_df.sample(n=1000, weights='target')
    train_df['vendor_description_clean'] = train_df['PROD_DESC_BY_VENDOR'].apply(strip_nonsense)

    # Query Walmart Search API using cleaned up vendor descriptions
    vendor_descriptions = set(train_df['vendor_description_clean'])
    description_category_map = {k:None for k in vendor_descriptions}
    for i, query in enumerate(vendor_descriptions):
        description_category_map[query] = predict_sub_category(query,key)
        if i % 50 == 0:
            print(f'Done with {i} of {len(vendor_descriptions)}!')

    train_df['prediction'] = train_df['vendor_description_clean'].map(description_category_map)
    train_df['prediction_truncated'] = train_df['prediction'].str.split("/").apply(lambda x: x[-1])
    train_df[['PROD_DESC_BY_VENDOR',
              'vendor_description_clean',
              'SUB_CATEGORY',
              'prediction',
              'prediction_truncated']].to_csv(r'walmart_query_predictions.csv',index=False)
