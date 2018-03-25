from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode
import params


def learn(x, y, test_x):

    clf = RandomForestClassifier(n_jobs=-1,
                                 n_estimators=params.n_estimators_rf,
                                 max_depth=params.max_depth_rf, random_state=0,
                                 min_samples_split=params.min_samples_split_rf,
                                 min_samples_leaf=params.min_samples_leaf_rf,
                                 max_features=params.max_feature_rf,
                                 max_leaf_nodes=params.max_leaf_nodes_rf,
                                 criterion=params.criterion_rf,
                                 min_impurity_split=params.min_impurity_split_rf,
                                 class_weight=params.cw_rf).fit(x, y)

    prediction_list = clf.predict(test_x)
    prediction_list_prob = clf.predict_proba(test_x)

    return prediction_list, prediction_list_prob
