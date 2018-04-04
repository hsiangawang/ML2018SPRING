import os, sys
import pandas as pd
import numpy as np
from random import shuffle
import argparse
from math import log, floor


def sigmoid(z):
  res = 1/(1+np.exp(-z))
  return np.clip(res,1e-8,1-(1e-8))

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test



def train(X_train, Y_train):
    # Split a 10%-validation set from the training set
    #valid_set_percentage = 0.1
    #X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
    #print(X_train.shape)
    #print(Y_train.shape)
    
    # Gaussian distribution parameters
    train_data_size = X_train.shape[0]
    cnt1 = 0
    cnt2 = 0

    mu1 = np.zeros((123,))
    mu2 = np.zeros((123,))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            mu1 += X_train[i]
            cnt1 += 1
        else:
            mu2 += X_train[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((123,123))
    sigma2 = np.zeros((123,123))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
    N1 = cnt1
    N2 = cnt2

    #print(shared_sigma.shape)
    sigma_inverse = np.linalg.pinv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_test.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)

    #print(y.shape)
    f = open(sys.argv[6], 'w')
    f.write('id,label\n')
    for i in range(0, y_.shape[0]):
      f.write(str(i+1)+','+str(int(y_[i]))+'\n')
    f.close()
    return

    # Load feature and label
X_all = pd.read_csv(sys.argv[3]).as_matrix().astype('float')
Y_all = pd.read_csv(sys.argv[4],header=None).as_matrix().astype('float64')
X_test = pd.read_csv(sys.argv[5]).as_matrix().astype('float')
#X_all,Y_all,X_test = load_data(sys.argv[2],sys.argv[3],sys.argv[4])
    # Normalization
X_all, X_test = normalize(X_all, X_test)
train(X_all,Y_all)

    




