from scipy.stats import mode
import numpy
import xgboost
import params


def learn(x, y, test_x):

    # set sample weight

    weight_list = []
    for j in range(len(y)):
        if y[j] == 0:
            weight_list.append(params.weight_0_xgb)
        if y[j] == 1000:
            weight_list.append(params.weight_1000_xgb)
        if y[j] == 1500:
            weight_list.append(params.weight_1500_xgb)
        if y[j] == 2000:
            weight_list.append(params.weight_2000_xgb)

    clf = xgboost.XGBClassifier(objective="multi:softmax", max_depth=params.max_depth_xgb,
                               learning_rate=params.learning_rate_xgb, n_estimators=params.n_estimators_xgb,
                               colsample_bytree=params.colsample_bytree_xgb, subsample=params.subsample_xgb,
                               min_child_weight=params.min_child_weight_xgb, gamma=params.gamma_xgb,
                               seed=params.random_seed, reg_alpha=params.reg_alpha_xgb,
                               reg_lambda=params.reg_lambda_xgb).fit(numpy.asarray(x), numpy.asarray(y),
                                                                        numpy.asarray(weight_list))
    prediction_list = clf.predict(test_x)
    prediction_list_prob = clf.predict_proba(test_x)

    return prediction_list, prediction_list_prob
