import numpy as np
import math
import pandas as pd

def matrix_factorization(r, k, iters=2, threshold=50000, l=.001, g=.02):
    #Initialize vectors
    p = np.random.rand(len(r), k)
    q = np.random.rand(len(r[0]), k)
    
    r_users, r_items = r.nonzero()
    r_nonzero_index = zip(r_users, r_items)

    #Minimize using SGD
    for itr in range(iters):
        np.random.seed(itr)
        np.random.shuffle(r_nonzero_index)
        for u, i in r_nonzero_index:
            if r[u][i] == 0:
                continue
            e = r[u][i] - np.dot(p[u], q[i].T) #Calculate prediction error
            #Update parameters
            q[i] += g * (e*p[u] - l*q[i])
            p[u] += g * (e*q[i] - l*p[u])
        train_error = root_mean_squared_error(np.dot(p, q.T), base_file)
        test_error = root_mean_squared_error(np.dot(p, q.T), test_file)
        print 'RMSE train: {}, test: {}'.format(train_error, test_error)
    return np.dot(p, q.T)

def generate_base_model(r):
    r_model = np.zeros((len(r), len(r[0])))
    for i in range(len(r[0])):
        x = 0
        counter = 0
        for u in range(len(r)):
            if r[u][i] != 0:
                x += r[u][i]
                counter += 1
        for u in range(len(r)):
            r_model[u][i] = x/counter if counter != 0 else 0
    return r_model

def root_mean_squared_error(train_R, test_file):
    error = 0
    counter = 0
    with open (test_file) as f:
        for line in f:
            counter += 1
            (user, item, rating, timestamp) = line.split('\t')
            error += math.pow((train_R[int(user)-1][int(item)-1] - int(rating)), 2)

        error = math.sqrt(error/counter)
    return error

error_list = []
base_error_list = []
k = 5
for i in range(k):
    base_file = 'ml-100k/u' + str(i+1) + '.base'
    test_file = 'ml-100k/u' + str(i+1) + '.test'

    #Loading data
    data = pd.read_csv(base_file, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    #Create User-Item dataframe
    user_item = data.groupby(['user_id', 'item_id'])['rating'].mean().unstack().fillna(0)
    #Test
    data_test = pd.read_csv(test_file, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    #Create User-Item dataframe
    user_item_test = data_test.groupby(['user_id', 'item_id'])['rating'].mean().unstack().fillna(0)

    user_item, _ = user_item.align(user_item_test, fill_value=0.0)

    user_item = user_item.values
    user_item_test = user_item_test.values
    r_hat = matrix_factorization(user_item, 10, iters=30, threshold=90000, l=1e-6, g=1e-3) #Calculate MF
    r_model = generate_base_model(user_item) #Generate evaluation about base line model of average for each item
    e = root_mean_squared_error(r_hat, test_file) #Evaluate MS with RMSE
    base_e = root_mean_squared_error(r_model, test_file) #Evaluate base model with RMSE
    error_list.append(e)
    base_error_list.append(base_e)
    print 'CV(MF) Fold{} error: {:.6f}'.format(i+1, e)
    print 'CV(base)  Fold{} error: {:.6f}'.format(i+1, base_e)

print 'CV(MF) mean: {}, std: {}'.format(np.mean(error_list), np.std(error_list))
print 'CV(base) mean: {}, std: {}'.format(np.mean(base_error_list), np.std(base_error_list))
