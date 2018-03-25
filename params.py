# print validation detail option
display_final_validation_result = True
display_part_validation_result = True

# set CV number
num_cv = 5
# set random seed, a given seed will generate specific random numbers
random_seed = 2017

######################################### gradient boosting parameters
n_estimators_gdbt = 200
learning_rate_gdbt = 0.1
max_depth_gdbt = 7
min_samples_split_gdbt = 200
min_samples_leaf_gdbt = 250
subsample_gdbt = 1.0
max_feature_gdbt = 'sqrt'

weight_0_gdbt = 1
weight_1000_gdbt = 40
weight_1500_gdbt = 50
weight_2000_gdbt = 90

########################################## Random Forest
n_estimators_rf = 750
max_depth_rf = 50
max_leaf_nodes_rf = None
min_samples_split_rf = 50
min_samples_leaf_rf = 20
max_feature_rf = 'sqrt'
criterion_rf = 'gini'
min_impurity_split_rf = 1e-7
bootstrap_rf = True

weight_0_rf = 4.40
weight_1000_rf = 59
weight_1500_rf = 111
weight_2000_rf = 130

cw_rf = {"0": weight_0_rf, "1000": weight_1000_rf, "1500": weight_1500_rf, "2000": weight_2000_rf}


#########################################  xgboosting parameters
n_estimators_xgb = 10  # 400
fearning_rate_xgb = 0.14
max_depth_xgb = 3
colsample_bytree_xgb = 0.09
subsample_xgb = 1
min_child_weight_xgb = 1
gamma_xgb = 0
reg_alpha_xgb = 0
reg_lambda_xgb = 1

weight_0_xgb = 2
weight_1000_xgb = 40  # 30
weight_1500_xgb = 60  # 60
weight_2000_xgb = 75  # 90
