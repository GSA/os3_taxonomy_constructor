# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 08:53:09 2018

@author: AustinPeel
"""

from os3_taxonomy_constructor import walmart 


office = walmart.getTaxonomyDF()
business =walmart.getTaxonomyDF(category='wallmart for business')    
walMartTaxonomy = office.append(business)
walMartTaxonomy.to_csv('data/wallMartTaxonomy.csv')




