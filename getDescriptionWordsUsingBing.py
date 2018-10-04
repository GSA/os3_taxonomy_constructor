# -*- coding: utf-8 -*-



from os3_taxonomy_constructor import bingAPI
import time, os, pandas as pd

wd = os.getcwd()

def getWebDataBySubCategory():
    df = _getTaxonomyData()
    for i in list(df.SUB_CATEGORY):
        search_results = bingAPI.getSearchResults(search_term= i)
        relatedSearches = bingAPI.getRelatedSearches(search_results)
        snippets = bingAPI.getWebPageSnippet(search_results)
        html =  bingAPI.getHTML(search_results)
        print("extracting " + i + " data from the web")
        time.sleep(1)
        result = {i:{'relatedSearches':relatedSearches,"snippets":snippets,"html":html}}
        df2 = _putResultsIntoDF(result)
        print("getting df")
        _saveDF(df2)
        print("saving df")


def _getTaxonomyData(LEVEL_1_CATEGORY = "OFFICE MANAGEMENT"):
    df = pd.read_csv(wd + "/data/taxonomy2.csv",encoding='iso-8859-1')
    df = df[df.LEVEL_1_CATEGORY == LEVEL_1_CATEGORY]
    return df 



def _putResultsIntoDF(results):
    df = pd.DataFrame(columns=['SUB_CATEGORY','text'])
    for i in results:
       related = " ".join(results[i]['relatedSearches'])
       snippets = " ".join(results[i]['snippets'])
       html = " ".join(results[i]['html'])
       allText =  related + " " + snippets +" " + html
       df = df.append(pd.DataFrame(data={'SUB_CATEGORY': [i], 'text': [allText]}))
    return df

def _saveDF(df):
    try:
        df2 = pd.read_csv(wd + "/data/bingSearchTaxonomy.csv")
        df2 = df2.append(df)
        df=df2
    except:
        print("no data to load")
    df.to_csv(wd + "/data/bingSearchTaxonomy.csv",index=False,)
    

if __name__ == "__main__":
    getWebDataBySubCategory()
