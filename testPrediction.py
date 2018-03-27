# -*- coding: utf-8 -*-
import pandas as pd
import os
import params
from models import gradient_boosting, random_forest

train_df = pd.read_csv('input/processed/train.csv')
test_df = pd.read_csv('input/processed/test.csv')
target = 'money'
test_ids = test_df['stu_id']
predictor = [x for x in train_df.columns if x not in [target, 'stu_id']]

X_train = train_df[predictor].values
y_train = train_df[target].values
X_test = test_df[predictor].values

model = 'gbdt'

if model == 'gbdt':
    predict_y_list, predict_y_prob_list = gradient_boosting.learn(X_train, y_train, X_test)
elif model == 'rf':
    predict_y_list, predict_y_prob_list = random_forest.learn(X_train, y_train, X_test)
elif model == 'xgb':
    predict_y_list, predict_y_prob_list = xgboosting.learn(X_train, y_train, X_test)

test_class_result = pd.DataFrame(columns=["studentid","subsidy"])
test_class_result.studentid = test_ids
test_class_result.subsidy = predict_y_list
test_class_result.subsidy = test_class_result['subsidy'].astype(int)
test_class_result_savepath = 'output/test_class_' + model + '.csv'
test_class_result.to_csv(test_class_result_savepath, index=False)

test_probe_result = pd.DataFrame(columns=["studentid","sub0","sub1000","sub1500","sub2000"])
test_probe_result.studentid = test_ids
test_probe_result.sub0 = predict_y_prob_list[:, 0]
test_probe_result.sub1000 = predict_y_prob_list[:, 1]
test_probe_result.sub1500 = predict_y_prob_list[:, 2]
test_probe_result.sub2000 = predict_y_prob_list[:, 3]
test_probe_result_savepath = 'output/test_probe_' + model + '.csv'
test_probe_result.to_csv(test_probe_result_savepath, index=False)
