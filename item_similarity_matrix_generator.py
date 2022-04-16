### Name: Bilal Sedef
### Date: 09.04.2022
### Description: This is the main file for the project.
### Topic: Recommendation engine for the Trendyol Database


import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

##
# Load the data
n = 0
for items in pd.read_csv('resources/articles_with_key_words.csv', skiprows=n, chunksize=1000):
    n += 1000
    items = items
    count_vectorizer = CountVectorizer(stop_words='english')
    sparse_matrix = count_vectorizer.fit_transform(items['Key_words'])
    zero_matrix = sparse.csr_matrix((sparse_matrix.shape[0], sparse_matrix.shape[0]))
    zero_matrix = zero_matrix.tolil()
    for i in tqdm(range(sparse_matrix.shape[0])):
        for j in range(sparse_matrix.shape[0]):
            zero_matrix[i, j] = cosine_similarity(sparse_matrix[i].toarray(), sparse_matrix[j].toarray())[0][0]
    zero_matrix = pd.DataFrame(zero_matrix.toarray(), columns=items['article_id'], index=items['article_id'])
    zero_matrix.to_csv(f'resources/item_similarity' + str(n) + '.csv')
    if n == 106000:
        break





