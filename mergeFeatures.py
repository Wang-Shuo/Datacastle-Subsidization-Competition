# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time

print('start merging features...')
start_time = time.time()

# read label information
subsidy_train = pd.read_table('input/train/subsidy_train.txt', sep=',', header=None)
subsidy_train.columns = ['stu_id','money']
id_test = pd.read_table('input/test/studentID_test.txt', sep=',', header=None)
id_test.columns = ['stu_id']
id_test['money'] = np.nan
train_test = pd.concat([subsidy_train, id_test])
print('read label information done.')

# merge score feature
score_train_test = pd.read_csv("input/processed/score_train_test.csv")
train_test = pd.merge(train_test, score_train_test, how="left", on="stu_id")
train_test["school_id"] = train_test["school_id"].fillna(-1)
train_test = train_test.set_index("stu_id")
# handle missing school value
for index, row in train_test.iterrows():
    if row["school_id"] == -1:
        before = index - 1
        after = index + 1
        if before in train_test.index:
            train_test.loc[index, "school_id"] = train_test.loc[before, "school_id"]
        elif after in train_test.index:
            train_test.loc[index, "school_id"] = train_test.loc[after, "school_id"]

train_test = train_test.reset_index()

train_test["school_id"] = train_test["school_id"].replace(-1, method="pad")
train_test["school_id"] = train_test["school_id"].astype(int)
train_test = pd.get_dummies(train_test, columns=["school_id"], prefix="school")
train_test = train_test.drop("grade_rank", axis=1)
rand_rank_rate = np.random.random_sample()
train_test["rank_rate"][np.isnan(train_test["rank_rate"])] = rand_rank_rate
print('merge score feature done.')

# merge dorm feature
dorm_train_test = pd.read_csv("input/processed/dorm_train_test.csv")
train_test = pd.merge(train_test, dorm_train_test, how="left", on="stu_id")
train_test["in_count"].fillna(0, inplace=True)
train_test["out_count"].fillna(0, inplace=True)
print('merge dorm feature done.')

# merge library feature
lib_io_count = pd.read_csv("input/processed/lib_io_count.csv")
train_test = pd.merge(train_test, lib_io_count, how="left", on="stu_id")
train_test["lib_io_count"].fillna(0, inplace=True)

borrow_book_count = pd.read_csv("input/processed/borrow_book_count.csv")
train_test = pd.merge(train_test, borrow_book_count, how="left", on="stu_id")
train_test["books_borrow_count"].fillna(0, inplace=True)
print('merge library feature done.')

# merge consume features
## consume by id
consume_by_id = pd.read_csv("input/processed/consume_by_id.csv")
train_test = pd.merge(train_test, consume_by_id, how="left", on="stu_id")
for col in list(train_test.columns[-7:]):
    train_test[col] = train_test[col].fillna(0)

# consume by type
consume_by_type = pd.read_csv("input/processed/consume_by_type.csv")
train_test = pd.merge(train_test, consume_by_type, how="left", on="stu_id")
for col in list(train_test.columns[-24:]):
    train_test[col] = train_test[col].fillna(0)

# consume by month
consume_by_month = pd.read_csv("input/processed/consume_by_month.csv")
train_test = pd.merge(train_test, consume_by_month, how="left", on="stu_id")
for col in list(train_test.columns[-5:]):
    train_test[col] = train_test[col].fillna(0)

# consume on summer vacation
consume_july_august = pd.read_csv("input/processed/consume_july_august.csv")
train_test = pd.merge(train_test, consume_july_august, how="left", on="stu_id")
for col in list(train_test.columns[-4:]):
    train_test[col] = train_test[col].fillna(0)

# consume by weekend
consume_by_weekend = pd.read_csv("input/processed/consume_by_weekend.csv")
train_test = pd.merge(train_test, consume_by_weekend, how="left", on="stu_id")
for col in list(train_test.columns[-4:]):
    train_test[col] = train_test[col].fillna(0)

# consume_interest_dining
consume_interest_dining = pd.read_csv("input/processed/consume_interest_dining.csv")
train_test = pd.merge(train_test, consume_interest_dining, how="left", on="stu_id")
for col in list(train_test.columns[-68:]):
    train_test[col] = train_test[col].fillna(0)

# consume interest supermarket
consume_interest_super = pd.read_csv("input/processed/consume_interest_super.csv")
train_test = pd.merge(train_test, consume_interest_super, how="left", on="stu_id")
for col in list(train_test.columns[-10:]):
    train_test[col] = train_test[col].fillna(0)

# consume by hour
consume_by_hour = pd.read_csv("input/processed/consume_by_hour.csv")
train_test = pd.merge(train_test, consume_by_hour, how="left", on="stu_id")
for col in list(train_test.columns[-96:]):
    train_test[col] = train_test[col].fillna(0)

print('merge consume feature done.')

# split train and test data, save them as csv file
train = train_test[train_test['money'].notnull()]
test = train_test[train_test['money'].isnull()]
train.to_csv("input/processed/train.csv", index=False)
test.to_csv("input/processed/test.csv", index=False)
print('saved merged features to file. total time: {}'.format(time.time() - start_time))
