# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 20:52:16 2016

@author: zhongda
"""

#To run this code, make sure you installed the right package, you can use pip or conda install to install packages
## import libraries
from datetime import datetime
import numpy as np
np.random.seed(123)
start=datetime.now()
import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers import Merge
import keras
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.callbacks import EarlyStopping

## Batch generators With this we can run code with multiple CPU ##################################################################################################################################

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch1 = X[batch_index,0:1139].toarray()
        X_batch2 = X[batch_index,1139:1153].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield [X_batch1, X_batch2], y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        X_batch1 = X[batch_index,:1139].toarray()
        X_batch2 = X[batch_index,1139:1153].toarray()
        counter += 1
        yield [X_batch1, X_batch2]
        if (counter == number_of_batches):
            counter = 0

########################################################################################################################################################

## read data
train = pd.read_csv('./train.csv')
test = pd.read_csv('./train.csv')


index = list(train.index)
print index[0:10]
np.random.shuffle(index)
print index[0:10]
train = train.iloc[index]
'train = train.iloc[np.random.permutation(len(train))]'

## set test loss to NaN
test['loss'] = np.nan

## response and IDs
y = np.log(train['loss'].values+200)
id_train = train['id'].values
id_test = test['id'].values

## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)

## Preprocessing and transforming to sparse data
sparse_data = []

f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data, tmp)


# 
#==============================================================================
## neural net
def nn_model():
    
    top_branch=Sequential()
    top_branch.add(Dense(450, input_dim= xtrain.shape[1]-14, init='he_normal',activation='linear'))
    top_branch.add(ELU(alpha=1.0))
    top_branch.add(BatchNormalization())
    top_branch.add(Dropout(0.2))
    
    bot_branch=Sequential()
    bot_branch.add(Dense(450, input_dim= 14, init='he_normal',activation='linear'))
    bot_branch.add(PReLU())
    bot_branch.add(BatchNormalization())
    bot_branch.add(Dropout(0.2))
    
    merged=Merge([top_branch,bot_branch],mode='mul')
    
    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(200,input_dim=200,init='he_normal'))
    final_model.add(PReLU())
    final_model.add(Dropout(0.4))
    final_model.add(BatchNormalization())
    final_model.add(Dense(1, init='he_normal'))
    
    final_model.compile(loss='mae',optimizer='adadelta')
    return(final_model)
    

## cv-folds
nfolds = 10
folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)

## train models
i = 0
nbags = 10
nepochs = 55
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

     

for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    pred = np.zeros(xte.shape[0])
    
    for j in range(nbags):
        model = nn_model()

        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 200, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0],
                                   verbose=0)
        pred +=np.exp(model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0])-200
        pred_test += np.exp(model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0])-200
         
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(yte, pred)
    i += 1
    print('Fold ', i, '- MAE:', score)

print('Total - MAE:', mean_absolute_error(np.exp(y)-200, pred_oob))

## train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
df.to_csv('preds_oob2.csv', index = False)

## test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv('submission2_keras.csv', index = False)


print datetime.now()-start