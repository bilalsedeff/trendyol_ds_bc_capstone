### Name: Bilal Sedef
### Date: 09.04.2022
### Description: This is the main file for the project.
### Topic: Recommendation engine for the Trendyol Database


import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm import trange
from scipy.sparse import csr_matrix
from collections import Counter
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

##
# Let's examine the transactions dataset

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
# Item-item similarity matrix creation is done.
########################################################################################################################

##
ladieswear_sim_mat = pd.read_csv('resources/item_similarity_for_ladieswear.csv', index_col=0)

##
# We are going to combine two different recommendation systems. One is based on the item-item similarity matrix
# and the other is based on matrix factorization. We are going to find the best combination of these two systems by tuning their contribution
# to overall recommendation accuracy.


########################################### Content Based Recommendation ###############################################


# We should find top K similar items to the items in our dataset
k = 11
k_matrix = np.zeros((len(all_articles), k))
k_matrix = pd.DataFrame(k_matrix, dtype=int)
k_matrix = k_matrix.set_index(all_articles)

##
for i, _ in tqdm(enumerate(all_articles)):
    top_k_similar_items = ladieswear_sim_mat.iloc[i, :].sort_values(ascending=False)[:k].index
    k_matrix.iloc[i, :] = top_k_similar_items

##
k_matrix = k_matrix.drop(columns=k_matrix.columns[0])

##
train_matrix_masked_knn = train_matrix_masked.copy()
##
# Content based recommendation
for i in tqdm(train_matrix_masked_knn.index):
    top_n = 20
    top_n_basket = []
    j = train_matrix_masked.loc[i, :]
    j = j[j > 0]
    j = j.index.tolist()
    for item in j:
        top_n_basket.extend([k_matrix.loc[item, :].tolist()][0])
    # Count the number of times each item appears in the basket
    count_dict = Counter(top_n_basket)
    # Get the top n items
    top_n_basket = count_dict.most_common(top_n)
    # Picking the ones occured more than once
    top_n_basket = [i[0] for i in top_n_basket if i[1] >= 2]
    # Get rid of quotation marks
    top_n_basket = [int(i) for i in top_n_basket]
    # Recommend the user the top n items
    train_matrix_masked_knn.loc[i, top_n_basket] = 1

# Calculate the error
known_error = 0
all_error = 0
for i, j in zip(masked_irow, masked_jcol):
    known_error += abs(train_matrix_unmasked.iloc[i, j] - train_matrix_masked_knn.iloc[i, j])
for i in train_matrix_masked_knn.index:
    all_error += abs(train_matrix_unmasked.loc[i, :].sum() - train_matrix_masked_knn.loc[i, :].sum())

print(f'Known error: {known_error}')
print(f'All error: {all_error}')

########################################### Matrix Factorization #######################################################


train_matrix_masked_mf = train_matrix_masked.copy()

##
# initialize factor matrices
d = 10
U = np.random.rand(train_matrix_masked_mf.shape[0], d) - 0.5
V = np.random.rand(d, train_matrix_masked_mf.shape[1]) - 0.5

##
irow_, jcol_ = np.where(train_matrix_ == 0)
idx = np.random.choice(np.arange(11100000), 15000, replace=False)
irow_ = irow_[idx]
jcol_ = jcol_[idx]

##
# Sum the rows and columns
summed_rows = np.concatenate((irow, irow_))
summed_cols = np.concatenate((jcol, jcol_))

##
# Stochastic Gradient descent
alpha = 0.03
my_lambda = 0.1
n_iters = 100

with trange(n_iters) as t:
    for _ in t:
        total_error = 0
        for i, j in zip(summed_rows, summed_cols):
            # Prediction
            y_pred = np.dot(U[i, :], V[:, j])
            # Error
            error = train_matrix_unmasked.iloc[i, j] - y_pred
            # Update
            U[i, :] += (2 * error * V[:, j] - 2 * my_lambda * U[i, :]) * alpha
            V[:, j] += (2 * error * U[i, :] - 2 * my_lambda * V[:, j]) * alpha

            total_error += error ** 2
            t.set_description(f'Total error: {total_error}')
        print(f'Total error: {total_error}')

##
# Constuct the recommendation matrix
mf_columns = train_matrix_masked_mf.columns.tolist()
mf_rows = train_matrix_masked_mf.index.tolist()

##
U_mf = pd.DataFrame(U, index=mf_rows, columns=range(d))
V_mf = pd.DataFrame(V, index=range(d), columns=mf_columns)

##
train_matrix_masked_mf_ = train_matrix_masked_mf.copy()
for i in tqdm(train_matrix_masked_mf_.index):
    for j in train_matrix_masked_mf_.columns:
        if train_matrix_masked_mf_.loc[i, j] == 0:
            pred = np.dot(U_mf.loc[i, :], V_mf.loc[:, j])
            if pred > 0.9:
                train_matrix_masked_mf_.loc[i, j] = 1
            else:
                train_matrix_masked_mf_.loc[i, j] = 0

##
train_matrix_masked_mf_.to_csv('train_matrix_masked_mf.csv')

##
train_matrix_masked_mf_ = pd.read_csv('resources/train_matrix_masked_mf.csv', index_col=0)
##
# Calculate the error
known_error = 0
all_error = 0
for i, j in tqdm(zip(summed_rows, summed_cols)):
    known_error += abs(train_matrix_unmasked.iloc[i, j] - train_matrix_masked_mf_.iloc[i, j])
for i in tqdm(train_matrix_masked_mf_.index):
    all_error += abs(train_matrix_unmasked.loc[i, :].sum() - train_matrix_masked_mf_.loc[i, :].sum())

print(f'Known error: {known_error}')
print(f'All error: {all_error}')


########################################################################################################################


########################################### Alternating Least Squares ##################################################

def calc_error(X, u_factors, i_factors):
    error = 0
    irows, jcols = np.where(X == 1)
    for i, j in tqdm(zip(irows, jcols)):
        error += abs(X.iloc[i, j] - np.dot(u_factors.iloc[i, :], i_factors.iloc[:, j]))
    return error


##
train_matrix_als = train_matrix_masked.copy()
##
# initialize factor matrices
n_factors = 5
n_items = train_matrix_als.shape[1]
n_users = train_matrix_als.shape[0]
# Unique items
items = train_matrix_als.columns.tolist()
# Unique users
users = train_matrix_als.index.tolist()

##

Q = pd.DataFrame(np.random.rand(n_factors, n_items) - 0.5, columns=items)
P = pd.DataFrame(np.random.rand(n_factors, n_users) - 0.5, columns=users)

##

X_train, X_test = train_test_split(train_matrix_als, test_size=0.1)

##
train_users = X_train.index.unique()
train_items = X_train.columns.unique()

##
R = csr_matrix(train_matrix_als.loc[train_users, train_items])

##

alpha = 0.030
my_lambda = 0.1
n_iters = 100

for t in tqdm(range(n_iters)):
    for u in train_users:
        I_u = X_train.loc[u, :].index.tolist()
        A = np.dot(Q.loc[:, I_u], Q.loc[:, I_u].T) + my_lambda * np.identity(n_factors)
        V = np.dot(Q.loc[:, I_u], R[np.where(u), np.where(I_u)].todense().T)
        P[u] = np.dot(np.linalg.inv(A), V)
    for i in train_items:
        U_i = X_train.loc[:, i].index.tolist()
        A = np.dot(P.loc[:, U_i], P.loc[:, U_i].T) + my_lambda * np.identity(n_factors)
        V = np.dot(P.loc[:, U_i], R[np.where(U_i), np.where(i)].todense().T)
        Q[i] = np.dot(np.linalg.inv(A), V)

    print("Iteration ", t)
    print("Train error: ", calc_error(X_train, P.T, Q))
    print("Test error: ", calc_error(X_test, P.T, Q))


##
# Building the recommendation matrix

for i in tqdm(train_matrix_als.index):
    for j in train_matrix_als.columns:
        if train_matrix_als.loc[i, j] == 0:
            pred = np.dot(P.loc[:, i], Q.loc[:, j])
            if pred > 0.9:
                train_matrix_als.loc[i, j] = 1
            else:
                train_matrix_als.loc[i, j] = 0

##
train_matrix_masked_als = train_matrix_als.copy()
##
articles_with_key_words = pd.read_csv('resources/articles_with_key_words.csv', index_col=0)
##
############################################## Ensemble ################################################################

iter = 100
w = np.random.rand(3, 1) - 0.5

R1 = train_matrix_masked_knn
R2 = train_matrix_masked_mf_
R3 = train_matrix_masked_als

# SGD for weights

for i in tqdm(range(iter)):
    error = 0
    for j in train_matrix_masked.index:
        for k in train_matrix_masked.columns:
            if train_matrix_masked.loc[j, k] == 0:
                pred = w[0] * R1.loc[j, k] + w[1] * R2.loc[j, k] + w[2] * R3.loc[j, k]
                if pred > 0.9:
                    train_matrix_masked.loc[j, k] = 1
                else:
                    train_matrix_masked.loc[j, k] = 0
                error += abs(train_matrix_masked.loc[j, k] - train_matrix_unmasked.loc[j, k])
                w += 0.01 * (train_matrix_unmasked.loc[j, k] - pred) * (R1.loc[j, k], R2.loc[j, k], R3.loc[j, k])



################################################ Not completed #########################################################
