import numpy as np
import pickle
import pandas as pd
import csv


def get_training_set(filename):
    ans = []
    with open(filename, 'rb') as csvfile:
        # csv_reader = pd.read_csv(filename, sep = '\n', delimiter=',')
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            ans.append([int(row['cat']), int(row['slave']), int(row['weekday']), int(row['food']), float(row['pred'])])
    return ans

def get_train_pk():
    train_file = '../data/training.csv'
    train = get_training_set(train_file)
    X = []
    y = []
    for each in train:
        X.append(each[0:4])
        y.append(each[4])
    with open('../data/training.pk1', 'wb') as f:
        pickle.dump(X, f)
        pickle.dump(y, f)
    return
def get_test_pk():
    example_file = '../data/sample.csv'
    sample = get_training_set(example_file)
    X = []
    y = []
    for each in sample:
        X.append(each[0:4])
    with open('../data/sample.pk1', 'wb') as f:
        pickle.dump(X, f)
    return

if __name__ == '__main__':
    get_test_pk()
    get_train_pk()
