# -*- coding: utf-8 -*-
import requests, re
from bs4 import BeautifulSoup


subscription_key = "key"
assert subscription_key
search_url = "https://api.cognitive.microsoft.com/bing/v7.0/search"



def getSearchResults(search_term= "air filters"):
    headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
    params  = {"q": search_term, "textDecorations":True, "textFormat":"HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    return search_results

def getWebPageSnippet(search_results):
    webpageSnippets = []
    try:
        for i in range(len(search_results['webPages']['value'])):
            snippet = search_results['webPages']['value'][i]['snippet']
            webpageSnippets.append(snippet)
    except:
        pass
    return webpageSnippets

def getRelatedSearches(search_results):
    texts = []
    try:
        for i in range(len(search_results['relatedSearches']['value'])):  
            text = search_results['relatedSearches']['value'][i]['text']
            texts.append(text)
    except:
        pass
    return texts

def getHTML(search_results):
    htmls = []
    for i in range(len(search_results['webPages']['value'])):
            try:    
                url = search_results['webPages']['value'][i]['url']
                print("going to" + url)
                r= requests.get(url)
                html = r.content
                cleanHTML = _cleanHTML(html)
                htmls.append(cleanHTML)
            except:
                continue
    return htmls

def _cleanHTML(html):        
    soup = BeautifulSoup(html) 
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    # get text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    # drop blank lines
    text = ' '.join([line.strip() for line in text.strip().splitlines()])
    text2 = re.sub(r'\W+', '', text)
    return text2
