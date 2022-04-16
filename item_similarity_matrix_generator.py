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
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

transactions = pd.read_csv('resources/transactions.csv')

##
# We are going to use last 2 days of data for test and last 7 days for training
transaction_dates = pd.to_datetime(transactions['t_dat'], format='%Y-%m-%d %H:%M:%S')
transaction_dates = transaction_dates.sort_values()[::-1]
last_date = transaction_dates.iloc[0]
threshold_date = last_date - pd.Timedelta(days=10)

##
start_date = last_date - pd.Timedelta(days=50)

##
# Train dates
transaction_train = transactions[pd.to_datetime(transactions['t_dat'], format='%Y-%m-%d %H:%M:%S') >= start_date]

##
# Let's load the products dataset
products = pd.read_csv('resources/articles.csv')

##
# Let's examine further the products dataset
print(products.head())
print(products.info())
print(products.describe())
print(products['index_name'].value_counts())

##
# We are going to start a ladies wear business.
# So let's look at how many people buying Ladies Wears
# First, we have to add a new column to the transactions dataset

a = np.asarray(products[products['index_name'].str.contains('Ladieswear')]['article_id'])
transaction_train['is_ladies_wear'] = transaction_train['article_id'].isin(a)

##
# Selecting only the ladies wear transactions

transaction_train = transaction_train[transaction_train['is_ladies_wear'] == True]

##
# We are going to use customers who has bought at least 5 ladies wear
cust_transc_count = transaction_train['customer_id'].value_counts() > 20
cust_transc_count = cust_transc_count[cust_transc_count == True]
scoped_customers = transaction_train[transaction_train['customer_id'].isin(cust_transc_count.index)]

##
transaction_train = scoped_customers

##
# Now we are creating our test dataset
transaction_test = transaction_train[
    pd.to_datetime(transaction_train['t_dat'], format='%Y-%m-%d %H:%M:%S') > threshold_date]

##
# We are going to subtract the test set from the train set
transaction_train = transaction_train[
    pd.to_datetime(transaction_train['t_dat'], format='%Y-%m-%d %H:%M:%S') <= threshold_date]

##
# We are going to use 20% of the train set for validation
transaction_train, transaction_val = train_test_split(transaction_train, test_size=0.2, random_state=42)

##
# Let's pull the unique names of the products and customers
customer_names_train = transaction_train['customer_id'].unique()
article_names_train = transaction_train['article_id'].unique()
customer_names_test = transaction_test['customer_id'].unique()
article_names_test = transaction_test['article_id'].unique()
customer_names_val = transaction_val['customer_id'].unique()
article_names_val = transaction_val['article_id'].unique()

##
# We are going to shuffle the sets
customer_names_train = np.random.permutation(customer_names_train)
article_names_train = np.random.permutation(article_names_train)
customer_names_test = np.random.permutation(customer_names_test)
article_names_test = np.random.permutation(article_names_test)
customer_names_val = np.random.permutation(customer_names_val)
article_names_val = np.random.permutation(article_names_val)

##
# We are going to create a sparse matrixes for the train, test and validation sets
# We are going to use the following parameters:
# - customer_names: the number of customers
# - article_names: the number of articles
# - transaction_train: the train set
# - transaction_val: the validation set
# - transaction_test: the test set

train_matrix = sparse.lil_matrix((len(customer_names_train), len(article_names_train)))
val_matrix = sparse.lil_matrix((len(customer_names_val), len(article_names_val)))
test_matrix = sparse.lil_matrix((len(customer_names_test), len(article_names_test)))

##
# We are going to put 1 for every transaction in the train set
for i, row in tqdm(transaction_train.iterrows(), total=len(transaction_train)):
    customer_id = np.where(customer_names_train == row['customer_id'])[0][0]
    article_id = np.where(article_names_train == row['article_id'])[0][0]
    train_matrix[customer_id, article_id] = 1

##
# We are going to put 1 for every transaction in the validation set
for i, row in tqdm(transaction_val.iterrows(), total=len(transaction_val)):
    customer_id = np.where(customer_names_val == row['customer_id'])[0][0]
    article_id = np.where(article_names_val == row['article_id'])[0][0]
    val_matrix[customer_id, article_id] = 1

##
# We are going to put 1 for every transaction in the test set
for i, row in tqdm(transaction_test.iterrows(), total=len(transaction_test)):
    customer_id = np.where(customer_names_test == row['customer_id'])[0][0]
    article_id = np.where(article_names_test == row['article_id'])[0][0]
    test_matrix[customer_id, article_id] = 1

##
# We are going to mask some of the non-zero values in the train matrix
train_matrix_ = pd.DataFrame(train_matrix.todense())
train_matrix_.columns = article_names_train
train_matrix_.index = customer_names_train

##
irow, jcol = np.where(train_matrix_ > 0)

##
idx = np.random.choice(np.arange(29500), 5000, replace=False)
masked_irow = irow[idx]
masked_jcol = jcol[idx]

##
train_matrix_unmasked = train_matrix_.copy()

##
for i, j in tqdm(zip(masked_irow, masked_jcol)):
    train_matrix_.iloc[i, j] = 0

##
train_matrix_masked = train_matrix_.copy()

##
# To csv both matrixes
train_matrix_masked.to_csv('resources/train_matrix_masked.csv')
train_matrix_unmasked.to_csv('resources/train_matrix_unmasked.csv')

##
########################################################################################################################
# We are going to create an item similarity matrix for our train set
print(len(article_names_train))
print(len(article_names_test))
print(len(article_names_val))

all_articles = np.concatenate((article_names_train, article_names_test, article_names_val))
all_articles = np.unique(all_articles)
print(len(all_articles))

##
item_sims = pd.read_csv('resources/articles_with_key_words.csv')
item_sims = item_sims.set_index('article_id')
item_sims = item_sims[item_sims.index.isin(all_articles)]

##
count_vectorizer = CountVectorizer(stop_words='english')
sparse_matrix = count_vectorizer.fit_transform(item_sims['Key_words'])
zero_matrix_shape = (all_articles.shape[0], all_articles.shape[0])
zero_matrix = csr_matrix(np.zeros(zero_matrix_shape), dtype=int)
zero_matrix = zero_matrix.tolil()
print(zero_matrix.shape)

##
# Converting zero matrix to dataframe
column_names = all_articles.tolist()
index_names = all_articles.tolist()
zero_matrix = pd.DataFrame(zero_matrix.toarray())

##
zero_matrix.columns = column_names
zero_matrix.index = index_names

##
cos_mat = cosine_similarity(sparse_matrix.toarray(), sparse_matrix.toarray())

##
# We are going to merge the cosine similarity matrix with the zero matrix we built.
zero_matrix.iloc[:, :] = cos_mat

##
zero_matrix.to_csv(f'resources/item_similarity_for_ladieswear.csv')
