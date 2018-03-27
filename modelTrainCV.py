# -*- coding: utf-8 -*-
import time
from sklearn.model_selection import StratifiedKFold
import pandas as pd

import params
from models import gradient_boosting, random_forest, xgboosting

def validate(predict_y_list, actual_y_list):

    num_right_dict = {}
    num_predict_dict = {}
    num_actual_dict = {}

    for (p_y, a_y) in zip(predict_y_list, actual_y_list):
        if p_y not in num_predict_dict:
            num_predict_dict[p_y] = 0
        num_predict_dict[p_y] += 1

        if a_y not in num_actual_dict:
            num_actual_dict[a_y] = 0
        num_actual_dict[a_y] += 1

        if p_y == a_y:
            if p_y not in num_right_dict:
                num_right_dict[p_y] = 0
            num_right_dict[p_y] += 1

    return num_right_dict, num_predict_dict, num_actual_dict



def calcF1(num_right, num_predict, num_actual):
    if num_predict == 0:
        precise = 0
    else:
        precise = float(num_right) / num_predict

    recall = float(num_right) / num_actual

    if precise + recall != 0:
        F1 = (2 * precise * recall) / (precise + recall)
    else:
        F1 = 0

    return F1


def run_model(X, y, model):

    avg_macro_f1 = 0
    num_total_samples = len(y)
    num_right_dict = {}
    num_predict_dict = {}
    num_actual_dict = {}

    for subsidy in [1000, 1500, 2000]:
        num_right_dict[subsidy] = 0
        num_predict_dict[subsidy] = 0
        num_actual_dict[subsidy] = 0

    # the txt file for saving cv results
    w = open('output/' + "train_class_" + model + ".txt", 'w')

    skf = StratifiedKFold(n_splits=params.num_cv, shuffle=True, random_state=params.random_seed)
    for train_idx, valid_idx in skf.split(X, y):
        start = time.time()

        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        if model == 'gbdt':
            predict_y_list, predict_y_prob_list = gradient_boosting.learn(X_train, y_train, X_valid)
        elif model == 'rf':
            predict_y_list, predict_y_prob_list = random_forest.learn(X_train, y_train, X_valid)
        elif model == 'xgb':
            predict_y_list, predict_y_prob_list = xgboosting.learn(X_train, y_train, X_valid)

        for label, prediction in zip(y_valid, predict_y_list):
            w.write(str(label) + ',' + str(prediction) + '\n')

        num_right_dict_part, num_predict_dict_part, num_actual_dict_part = validate(predict_y_list, y_valid)

        macro_f1_part = 0
        num_total_part = len(y_valid)
        for subsidy in [1000, 1500, 2000]:
            num_right_part = num_right_dict_part.get(subsidy, 0)
            num_predict_part = num_predict_dict_part.get(subsidy, 0)
            num_actual_part = num_actual_dict_part.get(subsidy, 0)
            F1_part = calcF1(num_right_part, num_predict_part, num_actual_part)
            macro_f1_part += float(num_actual_part) / num_total_part * F1_part
            if params.display_part_validation_result:
                print(str(subsidy) + "case :" + "   num_right: " + str(num_right_part) +  \
                                           "   num_predict: " + str(num_predict_part) + \
                                           "   num_actual: " + str(num_actual_part) + \
                                           "   f1: " + str(F1_part))

            num_right_dict[subsidy] += num_right_part
            num_predict_dict[subsidy] += num_predict_part
            num_actual_dict[subsidy] += num_actual_part

        print("macro F1: " + str(macro_f1_part))

        end = time.time()
        print('one cross validation done. time: ' + str(end - start))

    # calculate macro f1 score on total train data
    for subsidy in [1000, 1500, 2000]:
        num_right = num_right_dict.get(subsidy, 0)
        num_predict = num_predict_dict.get(subsidy, 0)
        num_actual = num_actual_dict.get(subsidy, 0)

        F1 = calcF1(num_right, num_predict, num_actual)
        avg_macro_f1 += float(num_actual) / num_total_samples * F1
        if params.display_final_validation_result:
            print(str(subsidy) + "case :" + "   num_right: " + str(num_right) +  \
                                       "   num_predict: " + str(num_predict) + \
                                       "   num_actual: " + str(num_actual) + \
                                       "   f1: " + str(F1))
    print('avg macro F1: ' + str(avg_macro_f1))

    w.close()

    return avg_macro_f1


if __name__ == '__main__':
    model = 'gbdt'

    train_df = pd.read_csv('input/processed/train.csv')
    target = 'money'
    predictor = [x for x in train_df.columns if x not in [target, 'stu_id']]
    X = train_df[predictor].values
    y = train_df[target].astype(int).values
    start = time.time()
    run_model(X, y, model)
    print('run model {} done. time: {}'.format(model, (time.time() - start)))
