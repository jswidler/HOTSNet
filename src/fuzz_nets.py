import numpy as np
import pandas as pd
import re
import random
import os
import time
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
import scipy.stats as stats

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Dropout, Dense, Activation, BatchNormalization, PReLU
from keras.optimizers import Adam

csv_data = pd.read_csv(os.path.join('data', 'hots_training_data.csv'), dtype=np.float32)

unplayed_maps = list()
for m in list(filter(lambda c: re.match("^map_",c) ,csv_data.columns)):
    if np.sum(csv_data[m]) == 0:
        unplayed_maps.append(m)
print("unplayed maps:", unplayed_maps)
unplayed_heroes = list()
for m in list(filter(lambda c: re.match("^[ab]_hero",c) ,csv_data.columns)):
    if np.sum(csv_data[m]) == 0:
        unplayed_heroes.append(m)
print("unplayed heroes:", unplayed_heroes)
csv_data = csv_data.drop(unplayed_maps + unplayed_heroes ,1)


winner = ['team_a_won']
gamemodes = list(filter(lambda c: re.match("^mode_",c), csv_data.columns))
print(gamemodes[:4])
herolevels = list(filter(lambda c: re.match("^[ab]_herolevel",c), csv_data.columns))
print(herolevels[:4])
herommrs = list(filter(lambda c: re.match("^[ab]_herommr",c), csv_data.columns))
print(herommrs[:4])
playerlevels = list(filter(lambda c: re.match("^[ab]_playerlevel",c), csv_data.columns))
print(playerlevels[:4])
playermmrs = list(filter(lambda c: re.match("^[ab]_playermmr",c), csv_data.columns))
print(playermmrs[:4])
maps = list(filter(lambda c: re.match("^map_",c), csv_data.columns))
print(maps[:4])
subgroups = list(filter(lambda c: re.match("^[ab]_subgroup_",c), csv_data.columns))
print(subgroups[:4])

print(len(winner+gamemodes+herolevels+herommrs+playerlevels+playermmrs+maps+subgroups), len(csv_data.columns))


# training_data, validation_data = train_test_split(csv_data, test_size=50000)
# training_data.index = np.arange(0, training_data.shape[0])
# validation_data.index = np.arange(0, validation_data.shape[0])


def split_features(data):
    labels = data['team_a_won'].astype(bool)
    features = data.drop(['team_a_won'],1)
    return features, labels

features, labels = split_features(csv_data)


class StatLog(Callback): 
    train_start = 0
    epoch_start = 0
    def on_train_begin(self, logs):
        self.train_start = time.time()
    
    def on_train_end(self, logs):
        n = time.time() - self.epoch_start
        seconds = int(time.time() - self.train_start + .5)
        print('Training round {}:{}'.format(seconds // 60, seconds % 60))
        
    def on_epoch_begin(self, epoch, logs):
        self.epoch_start = time.time()
        print("Epoch " + str(epoch+1), end='', flush=True)
        
    def on_epoch_end(self, epoch, logs={}):
        n = time.time() - self.epoch_start
        print("({}s) - train_loss: {:.5f}; train_acc: {:.3f}%; val_loss: {:.5f}; val_acc: {:.3f}%".format(
            int(time.time() - self.epoch_start + .5),
            logs['loss'], 100*logs['acc'], logs['val_loss'], 100*logs['val_acc']))
        


def hotnet_init(input_dim, xs=[500,500], ds=[.5,.5], lr=.001, beta_1=.9, activation='relu', norm=False):
    model = Sequential()
    
    a = 'linear' if activation is 'prelu' else activation
    
    model.add(Dense(xs[0], input_dim=input_dim, activation=a))
    if activation is 'prelu':
        model.add(PReLU())
    if ds[0] > 0:
        model.add(Dropout(ds[0]))
    
    for i in range(1, len(xs)):
        model.add(Dense(xs[i], activation=a))
        if activation is 'prelu':
            model.add(PReLU())
        model.add(Dropout(ds[i]))
    
    if norm:
        model.add(BatchNormalization())
        
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=beta_1), metrics=['accuracy'])
    return model

MODEL_FOLDER = 'saved_models'

best_models = [{
    'name': '1081',
    'xs': [1081], 'ds': [.4],
    'lr': 0.0001, 'beta_1': .9,
    'batch_size': 1687,
    'norm': False
},{
    'name': '1213x185',
    'xs': [1213, 185], 'ds': [0.5, 0.3],
    'lr': 0.0001, 'beta_1': 0.8,
    'batch_size': 3887,
    'norm': False
},{
    'name': '1090x737',
    'xs': [1090, 737], 'ds': [0.6, 0.5],
    'lr': 0.0003831772719601958, 'beta_1': 0.9,
    'batch_size': 741,
    'norm': True
},{
    'name': '1402x704',
    'xs': [1402, 704], 'ds': [0.5, 0.0],
    'lr': 0.0001, 'beta_1': 0.9,
    'batch_size': 2903,
    'norm': False
},{
    'name': 'random-qvtmacbk', 'xs': [1380], 'ds': [0.0], 'lr': .0000325499964238507, 'beta_1': 0.858558505030204, 'batch_size': 4873, 'norm': False
},{
    'name': 'random-iyuzxbhp', 'xs': [1014], 'ds': [0.40000000000000002], 'lr': 0.00021808771904722899, 'beta_1': 0.7628453007161774, 'batch_size': 4259, 'norm': False
},{
    'name': 'random-zvceymbi', 'xs': [437], 'ds': [0.59999999999999998], 'lr': .00007533878893017083, 'beta_1': 0.7211260661917976, 'batch_size': 1343, 'norm': False
},{
    'name': 'random-gxkwtwnx', 'xs': [742], 'ds': [0.40000000000000002], 'lr': .000035424951003431825, 'beta_1': 0.7178868876566145, 'batch_size': 1804, 'norm': False
}]

def hotnet_model(
    data, 
    target,
    epochs=50, kfold=10, 
    model_init=hotnet_init,
    model_file='amodel',
    lr=.001,
    beta_1=0.9,
    batch_size=10000
):  
    scores = []
    fold = StratifiedKFold(n_splits=kfold, shuffle=True)
    for i, (train, test) in enumerate(fold.split(data, target)):
        print('{} - Fold {}/{}'.format(model_file, i+1, kfold))
        filename = os.path.join(MODEL_FOLDER, "{}-{}.hdf5".format(model_file, i+1))
        model = model_init(data.shape[1])
        model.fit(
            data.loc[train].values, target.loc[train], 
            validation_data=(data.loc[test].values, target.loc[test]),
            epochs=epochs, batch_size=batch_size, 
            callbacks=[
                ModelCheckpoint(filepath=filename, verbose=0, save_best_only=True),
                StatLog(),
                EarlyStopping(monitor='val_loss', min_delta=.000002, patience=4, verbose=0)
            ], verbose=0)
        
        model.load_weights(filename)
        s = model.evaluate(data.loc[test].values, target[test], verbose=0)
        acc = 100*s[1]
        print("Test loss: {:.5f}; acc: {:.3f}%".format(s[0], acc))
        scores.append(acc)
        if acc == np.max(scores):
            best_model = filename
        if acc < 61.55:
            print('Looking for better candidates, skipping additional folds')
            break
    with open(os.path.join(MODEL_FOLDER, "{}.json".format(model_file)), 'w') as f:
        f.write(model.to_json())
    os.rename(best_model, os.path.join(MODEL_FOLDER, "{}-{:.4}.hdf5".format(model_file, np.mean(scores))))
    print("Model accuracy: {:.2f}% +/- {:.2f}%".format(np.mean(scores), np.std(scores)))
    return scores

def truncnorm(mu, sigma, lower, upper):
    return stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

beta_1_gen = truncnorm(.75, .08, .5, .95)
batch_size_gen = truncnorm(1000, 2000, 300, 5000)
lr_gen = truncnorm(.00025, .00013, .000005, .0005)
                   
def random_net():
    l_c = np.random.randint(1,3) # 1-2 layers
    return {
        'name': 'random-' + ''.join([np.random.choice(list('abcdefghijklmnopqrstuvwxyz')) for _ in range(0,8)]),
        'xs': [np.random.randint(300,1500) if i is 0 else np.random.randint(10,150) for i in range(0, l_c)],
        'ds': [np.random.choice([0, .3, .4, .5, .6]) for i in range(0, l_c)],
        'lr': lr_gen.rvs(),
        'beta_1': beta_1_gen.rvs(),
        'batch_size': int(batch_size_gen.rvs()),
        'norm': False
    }

def hotnet_initx(input_dim):
    return hotnet_init(input_dim, xs=xs, ds=ds, lr=lr, beta_1=beta_1, activation='prelu', norm=norm)

scores = {}
p = []

while True:
    params = random_net()
    p.append(params)
    with open(os.path.join(MODEL_FOLDER, "params"), 'w') as f:
        f.write(str(p))
    print('params', params)
    xs = params['xs']
    ds = params['ds']
    lr = params['lr']
    norm = params['norm']
    beta_1 = params['beta_1']
    scores[params['name']] = hotnet_model(
        features, labels,
        kfold=3, epochs=100, batch_size=params['batch_size'],
        model_file=params['name'], model_init=hotnet_initx, 
    )
    print(p)
    for l, s in scores.items():
        print("{}: {:.2f}% +/- {:.2f}%".format(l, np.mean(s), np.std(s)))
