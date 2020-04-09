#! python

# Based on script created by Aleksei Wan on 13.11.2019
# Adapted for use in this project by Aleksei Wan

# Imports
import os
import sys
import json
from argparse import ArgumentParser
import lightgbm as lgbm
import numpy as np


def train_lgbm_model(params, training_data, validation_data=None, save_model_path: str = 'lgbm-model.txt'):
    bst = lgbm.train(params, training_data, valid_sets=[validation_data])
    bst.save_model(save_model_path)
    return bst


def calculate_acc(predictions, labels):
    acc, round_acc = 0, 0
    for idx, val in enumerate(predictions):
        c = np.argsort(val)
        if np.argmax(val) == labels[idx]:
            acc += 1
        if c[1] == labels[idx] or c[2] == labels[idx]:
            round_acc += 1
    round_acc += acc
    acc /= (len(labels))
    round_acc /= len(labels)
    return acc, round_acc


def make_printable(x) -> str: return str(round(100 * float(x), 3))

def load_file(contents):
    data_list = []
    labels = []
    for element in contents:
        data_list.append(element[0])
        labels.append(element[1])
    return np.array(data_list), np.array(labels)

if __name__ == "__main__":
    parser = ArgumentParser(description='Create a LightGBM tree based on provided data')
    parser.add_argument('--training_file', '-t', type=str, default='trainresults.json', help='File containing training')
    parser.add_argument('--validation_file', '-v', type=str, default='valresults.json', help='File containing val')
    parser.add_argument('--epochs', '-e', type=int, default=1, help='File containing results')
    args = parser.parse_args()
    path_to_check = os.path.abspath(args.training_file)
    path_to_check2 = os.path.abspath(args.validation_file)
    if not os.path.exists(path_to_check) or not os.path.exists(path_to_check2):
        print('Provided path', path_to_check, 'is not a valid directory. Please try again')
        sys.exit(-1)

    # Read Input JSON file
    with open(path_to_check, 'r') as file:
        contents = json.load(file)
    train_data, train_labels = load_file(contents)
    with open(path_to_check2, 'r') as file:
        contents = json.load(file)
    valid_data, valid_labels = load_file(contents)
    with open(os.path.abspath('alexnet/testresults.json'), 'r') as file:
        contents = json.load(file)
    # Create data sets
    training_data = lgbm.Dataset(train_data, label=train_labels)
    validation_data = lgbm.Dataset(valid_data, label=valid_labels)

    # Run training
    acc_list = np.zeros((args.epochs, 3))
    param_list = []
    for i in range(args.epochs):
        param_list.append({'objective': 'multiclass',
             'num_class': 8,
             'metric': 'multi_logloss',
             'early_stopping_rounds': 100,
             'max_depth': 8,
             'max_bin': 255,
             'feature_fraction': 1,
             'bagging_fraction': 0.65,
             'bagging_freq': 5,
             'learning_rate': 0.020,
             'num_rounds': 1000,
             'num_leaves': 32,
             'min_data_in_leaf': 10,
             'lambda_l1': 0.0001})

        bst = train_lgbm_model(params=param_list[i], training_data=training_data, validation_data=validation_data,
                               save_model_path='lgbm-model_' + str(i) + '.txt')

        acc_list[i][0], acc_list[i][2] = calculate_acc(bst.predict(train_data), train_labels)
        acc_list[i][1], acc_list[i][3] = calculate_acc(bst.predict(valid_data), valid_labels)

    np.savetxt('lgbm_model_accuracies.csv', acc_list, delimiter=",", fmt='%s')
    with open('lgbm_model_params.txt', 'w') as outfile:
        json.dump(param_list, outfile)

    vt_acc_list = np.array([acc_list[i][1] for i in range(args.epochs)])

    best_tree_idx = np.argmax(vt_acc_list)
    print('Tree:', best_tree_idx, '(with validation accuracy', make_printable(vt_acc_list[best_tree_idx]) + '%)')
    print('Tree Params:', param_list[best_tree_idx], '\n')

    print('Training Samples:', len(train_labels))
    print('Validation Samples:', len(valid_labels))
