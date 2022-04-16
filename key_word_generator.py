### Name: Bilal Sedef
### Date: 09.04.2022
### Description: This is the main file for the project.
### Topic: Recommendation engine for the Trendyol Database


import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sklearn.metrics as metrics
from rake_nltk import Rake
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error

##
# Load the data
items = pd.read_csv('resources/articles.csv', )

##
print(items.columns)
##
items['Key_words'] = ''

for index, row in tqdm(items.iterrows()):
    r = Rake()
    r.extract_keywords_from_text(str(row['prod_name']) + ' ' + str(row['product_type_no'])
                                 + ' ' + str(row['product_type_name']) + ' ' + str(row['product_group_name'])
                                 + ' ' + str(row['graphical_appearance_name']) + ' ' + str(row['colour_group_code']) + ' ' + str(row['colour_group_name'])
                                 + ' ' + str(row['perceived_colour_value_id']) + ' ' + str(row['perceived_colour_value_name']) + ' ' + str(row['perceived_colour_master_id'])
                                 + ' ' + str(row['perceived_colour_master_name']) + ' ' + str(row['department_no']) + ' ' + str(row['department_name'])
                                 + ' ' + str(row['index_code']) + ' ' + str(row['index_name']) + ' ' + str(row['index_group_no']) + ' ' + str(row['index_group_name'])
                                 + ' ' + str(row['section_no']) + ' ' + str(row['section_name']) + ' ' + str(row['garment_group_no']) + ' ' + str(row['garment_group_name'])
                                 + ' ' + str(row['detail_desc']))

    key_words_dict_scores = r.get_word_degrees()
    row['Key_words'] = list(key_words_dict_scores.keys())
    row = str(row['Key_words'])
    row = row.replace('[', '').replace(']', '').replace("'", '').replace(' ', '').replace(',', ' ')
    items.loc[index, 'Key_words'] = row


##
# We are going to export our new dataframe to a csv file
items.to_csv('resources/articles_with_key_words.csv', index=False)

