from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import mode
import params

def learn(x, y, test_x):
    # set sample weight
    weight_list = []

    for j in range(len(y)):
        if y[j] == 0:
            weight_list.append(params.weight_0_gdbt)
        if y[j] == 1000:
            weight_list.append(params.weight_1000_gdbt)
        if y[j] == 1500:
            weight_list.append(params.weight_1500_gdbt)
        if y[j] == 2000:
            weight_list.append(params.weight_2000_gdbt)

    clf = GradientBoostingClassifier(loss='deviance', n_estimators=params.n_estimators_gdbt,
                                     learning_rate=params.learning_rate_gdbt,
                                     max_depth=params.max_depth_gdbt, random_state=params.random_seed,
                                     min_samples_split=params.min_samples_split_gdbt,
                                     min_samples_leaf=params.min_samples_leaf_gdbt,
                                     subsample=params.subsample_gdbt,
                                     max_features=params.max_feature_gdbt).fit(x, y, weight_list)
    prediction_list = clf.predict(test_x)
    prediction_list_prob = clf.predict_proba(test_x)

    return prediction_list, prediction_list_prob
