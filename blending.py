# -*- coding: utf-8 -*-
import params
import time
import pandas as pd


def prepare_blending_trainset(train_prediction_files_list):
    predictions_list = []
    for file in train_prediction_files_list:
        predictions_list.append(open(file).readlines())

    with open('output/blending_trainset.txt', 'w') as f:
        for i in range(len(predictions_list[0])):
            output_line = ""
            output_line += predictions_list[0][i].strip('\n') + ','

            for j in range(1, len(predictions_list)):
                output_line += predictions_list[j][i].strip('\n').split(',')[1] + ','

            output_line = output_line[:-1]
            f.write(output_line + '\n')


def prepare_blending_testset(test_prediction_files_list):
    predictions_df = pd.read_csv(test_prediction_files_list[0])
    for file in test_prediction_files_list[1:]:
        df = pd.read_csv(file)
        predictions_df = pd.merge(predictions_df, df, on='studentid', how='left')

    predictions_df.to_csv('output/blending_testset.csv', index=False)


def vote(X, weight):
    result = []
    for predict_list in X:
        predict_dict = {}
        for i in range(len(predict_list)):
            if predict_list[i] in predict_dict:
                predict_dict[predict_list[i]] += weight[i]
            else:
                predict_dict[predict_list[i]] = weight[i]

        sorted_dict_list = sorted(predict_dict.items(), key=lambda d: d[1], reverse=True)
        result.append(int(sorted_dict_list[0][0]))

    return result


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


def run_blending_offline():
    avg_macro_f1 = 0
    num_right_dict = {}
    num_predict_dict = {}
    num_actual_dict = {}

    for subsidy in [1000, 1500, 2000]:
        num_right_dict[subsidy] = 0
        num_predict_dict[subsidy] = 0
        num_actual_dict[subsidy] = 0

    blending_df = pd.read_table('output/blending_trainset.txt', sep=',', header=None)
    blending_x = blending_df.iloc[:, 1:].values
    blending_y = blending_df.iloc[:, 0].values
    num_total_samples = len(blending_y)
    predict_y_list = vote(blending_x, params.voteWeight)

    num_right_dict, num_predict_dict, num_actual_dict = validate(predict_y_list, blending_y)
    for subsidy in [1000, 1500, 2000]:
        num_right = num_right_dict.get(subsidy, 0)
        num_predict = num_predict_dict.get(subsidy, 0)
        num_actual = num_actual_dict.get(subsidy, 0)

        F1 = calcF1(num_right, num_predict, num_actual)
        avg_macro_f1 += float(num_actual) / num_total_samples * F1
        print(str(subsidy) + "case :" + "   num_right: " + str(num_right) +  \
                                       "   num_predict: " + str(num_predict) + \
                                       "   num_actual: " + str(num_actual) + \
                                       "   f1: " + str(F1))
    print('avg macro F1: ' + str(avg_macro_f1))


def run_blending_online():
    blending_df = pd.read_csv('output/blending_testset.csv')
    blending_x = blending_df.iloc[:, 1:].values

    test_class_result = pd.DataFrame(columns=["studentid","subsidy"])
    test_class_result.studentid = blending_df['studentid']
    test_class_result.subsidy = vote(blending_x, params.voteWeight)
    test_class_result.subsidy = test_class_result['subsidy'].astype(int)
    test_class_result_savepath = 'output/test_blend.csv'
    test_class_result.to_csv(test_class_result_savepath, index=False)


if __name__ == '__main__':
    train_prediction_files_list = ['output/train_class_gbdt.txt', 'output/train_class_rf.txt', 'output/train_class_xgb.txt']
    test_prediction_files_list = ['output/test_class_gbdt.csv', 'output/test_class_rf.csv', 'output/test_class_xgb.csv']
    prepare_blending_trainset(train_prediction_files_list)
    run_blending_offline()
    prepare_blending_testset(test_prediction_files_list)
    run_blending_online()
