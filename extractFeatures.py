# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time

def ScoreProcess():
    """
    extract student score ranking percentage feature
    """
    if os.path.exists('input/processed/score_train_test.csv'):
        return

    score_train = pd.read_table('input/train/score_train.txt', sep=',', header=None)
    score_train.columns = ["stu_id","school_id","grade_rank"]
    score_test = pd.read_table('input/test/score_test.txt', sep=',', header=None)
    score_test.columns = ["stu_id","school_id","grade_rank"]
    score_train_test = pd.concat([score_train,score_test])

    #compute total people of every school
    school_dict = {}
    for sch_id in score_train_test["school_id"].unique():
        school_dict[sch_id] = score_train_test[score_train_test["school_id"] == sch_id]["grade_rank"].max()

    #compute students' rank rate at his/her school
    score_train_test["rank_rate"] = score_train_test.apply(lambda row : row["grade_rank"] / school_dict[row["school_id"]], axis=1)
    # save the processed score dataframe in csv format
    score_train_test.to_csv("input/processed/score_train_test.csv", index=False)


def DormProcess():
    """
    extract student's dorm in and out frequency
    """
    if os.path.exists('input/processed/dorm_train_test.csv'):
        return

    dorm_train = pd.read_table('input/train/dorm_train.txt', sep=',', header=None)
    dorm_train.columns = ["stu_id","when","direction"]
    dorm_test = pd.read_table('input/test/dorm_test.txt', sep=',', header=None)
    dorm_test.columns = ["stu_id","when","direction"]
    dorm_train_test = pd.concat([dorm_train,dorm_test])

    # count the frequency of in and out
    cleaned_dorm = pd.DataFrame({"in_count":dorm_train_test.groupby(["stu_id"])["direction"].sum(),
                                    "all_count":dorm_train_test.groupby(["stu_id"])["direction"].count()}).reset_index()

    cleaned_dorm["out_count"] = cleaned_dorm["all_count"] - cleaned_dorm["in_count"]
    cleaned_dorm.drop("all_count", axis=1, inplace=True)
    cleaned_dorm.to_csv("input/processed/dorm_train_test.csv", index=False)


def LibraryProcess():
    """
    extract student's library IO and book borrow frequency
    """
    if os.path.exists('input/processed/lib_io_count.csv') and os.path.exists('input/processed/borrow_book_count.csv'):
        return

    # library io data
    lib_train = pd.read_table('input/train/library_train.txt', sep=',', header=None)
    lib_train.columns = ["stu_id","gate","time"]
    lib_test = pd.read_table('input/test/library_test.txt', sep=',', header=None)
    lib_test.columns = ["stu_id","gate","time"]
    lib_train_test = pd.concat([lib_train,lib_test])

    lib_io_count = pd.DataFrame({"lib_io_count": lib_train_test.groupby(["stu_id"])["gate"].count()}).reset_index()
    lib_io_count.to_csv("input/processed/lib_io_count.csv", index=False)

    # process book borrow data
    def bookBorrowProcess(borrow_file):
        borrow_list = []
        with open(borrow_file, 'r', encoding='utf-8') as file:
            content = file.readlines()
            for line in content:
                borrow_dict = {}
                split_line = line.split(',')
                borrow_dict["stu_id"] = split_line[0]
                borrow_dict["borrow_time"] = split_line[1]
                borrow_dict["book_name"] = split_line[2]
                if split_line[-1] == split_line[2]:
                    borrow_dict["book_code"] = np.nan
                else:
                    borrow_dict["book_code"] = split_line[-1].rstrip('\n')
                borrow_list.append(borrow_dict)

        borrow_list_df = pd.DataFrame(borrow_list)
        return borrow_list_df

    borrow_train_df = bookBorrowProcess('input/train/borrow_train.txt')
    borrow_test_df = bookBorrowProcess('input/test/borrow_test.txt')
    borrow_train_test = pd.concat([borrow_train_df, borrow_test_df])
    borrow_book_count = pd.DataFrame({"books_borrow_count": borrow_train_test.groupby(["stu_id"])["book_name"].count()}).reset_index()
    borrow_book_count.to_csv("input/processed/borrow_book_count.csv", index=False)

def CardProcess():
    """
    extract multiple dimensional consume features from card data, including consume places, consume type, consume time.
    """
    # some preprocess on card data
    card_train = pd.read_table("input/train/card_train.txt", sep=",", header=None)
    card_train.columns = ["stu_id", "consume_form", "where", "type", "when", "amount", "residual"]
    card_test = pd.read_table("input/test/card_test.txt", sep=",", header=None)
    card_test.columns = ["stu_id", "consume_form", "where", "type", "when", "amount", "residual"]
    card_train_test = pd.concat([card_train, card_test])
    # filter the card data using consume_form column selecting “POS消费”，“nan”，“车载消费”
    filter_cond = (card_train_test["consume_form"]=="POS消费") | (card_train_test["consume_form"]=="车载消费") | (card_train_test["consume_form"].isnull())
    consume_train_test = card_train_test[filter_cond]
    consume_train_test = consume_train_test.drop(["consume_form"],axis=1)
    # convert Chinese into english in Series "type" using map function
    type_dict = {"食堂": 0, "超市": 1, "图书馆": 2 , "洗衣房": 3, "开水": 4, "淋浴": 5, "文印中心": 6, "教务处": 7, "校车": 8, "校医院": 9, "其他": 10}
    consume_train_test["type"] = consume_train_test["type"].map(type_dict)
    consume_train_test["type"].fillna(11, inplace=True)
    consume_train_test["type"] = consume_train_test["type"].astype(int)
    consume_train_test["where"] = consume_train_test["where"].str.extract("(\d+)")
    consume_train_test["where"].fillna(-1, inplace=True)
    consume_train_test.to_csv("input/processed/consume_train_test.csv", index=False)

    # feature 1: group the consume_train_test data by stu_id and compute some features like count, min, max, sum.
    consume_by_id = pd.DataFrame({"consume_count": consume_train_test.groupby(['stu_id'])['amount'].count(),
                                  "consume_sum": consume_train_test.groupby(['stu_id'])['amount'].sum(),
                                  "consume_max": consume_train_test.groupby(['stu_id'])['amount'].max(),
                                  "consume_median": consume_train_test.groupby(['stu_id'])['amount'].median()}).reset_index()

    consume_by_id["residual_sum"] = consume_train_test.groupby(['stu_id'])['residual'].sum()
    consume_by_id["residual_max"] = consume_train_test.groupby(['stu_id'])['residual'].max()
    consume_by_id["residual_median"] = consume_train_test.groupby(['stu_id'])['residual'].median()
    consume_by_id.to_csv("input/processed/consume_by_id.csv", index=False)


    # feature 2: extract some statistic features based on consume type
    # sum of different consume type
    consume_by_type_sum = consume_train_test.groupby(['stu_id', 'type'])["amount"].sum().unstack().reset_index()
    # some types have too many missing values, so drop them and keep the type 0,1,2,3,4,5,6,8
    consume_by_type_sum.drop([7, 9, 10, 11], axis=1, inplace=True)
    # change the column names which are more indicative
    consume_by_type_sum.columns = ['stu_id', 'type_0_sum', 'type_1_sum', 'type_2_sum', 'type_3_sum', 'type_4_sum', 'type_5_sum', 'type_6_sum', 'type_8_sum']
    # count of different consume type
    consume_by_type_count = consume_train_test.groupby(['stu_id', 'type'])["amount"].count().unstack().reset_index()
    consume_by_type_count.drop([7, 9, 10, 11], axis=1, inplace=True)
    consume_by_type_count.columns = ['stu_id', 'type_0_count', 'type_1_count', 'type_2_count', 'type_3_count', 'type_4_count', 'type_5_count', 'type_6_count', 'type_8_count']
    # max of different consume type
    consume_by_type_max = consume_train_test.groupby(['stu_id', 'type'])["amount"].max().unstack().reset_index()
    consume_by_type_max.drop([7, 9, 10, 11], axis=1, inplace=True)
    consume_by_type_max.columns = ['stu_id', 'type_0_max', 'type_1_max', 'type_2_max', 'type_3_max', 'type_4_max', 'type_5_max', 'type_6_max', 'type_8_max']
    # merge the consume_by_type data
    consume_by_type = pd.merge(consume_by_type_sum, consume_by_type_count, how='left', on='stu_id')
    consume_by_type = pd.merge(consume_by_type, consume_by_type_max, how='left', on='stu_id')
    consume_by_type.to_csv("input/processed/consume_by_type.csv", index=False)


    # feature 3: extract consume monthly of every student
    consume_train_test["when"] = consume_train_test.when.apply(lambda t: datetime.strptime(t,"%Y/%m/%d %H:%M:%S"))
    consume_train_test["year"] = consume_train_test.when.apply(lambda t: t.year)
    consume_train_test["month"] = consume_train_test.when.apply(lambda t: t.month)
    consume_train_test["day"] = consume_train_test.when.apply(lambda t: t.day)
    consume_train_test["hour"] = consume_train_test.when.apply(lambda t: t.hour)
    consume_train_test.drop(['residual'], axis=1, inplace=True)
    consume_train_test.to_csv("input/processed/consume_train_test_timesplit.csv", index=False)
    # monthly consume
    consume_monthly = pd.DataFrame({"days_in_a_month":consume_train_test.groupby(["stu_id","year","month"])["day"].count(),
                                "consume_by_month":consume_train_test.groupby(["stu_id","year","month"])["amount"].sum()}).reset_index()
    # rule out some vacation months, like summer and winter vacation. including 2014.1,2014.2,2014.7,2014.8,2015.1,2015.2,2015.7,2015.8
    for month in [1,2,7,8]:
        consume_monthly = consume_monthly[consume_monthly["month"] != month]
    # filter the month with a abnormal consume frequency, less than 3 in column "days_in_a_month"
    consume_monthly = consume_monthly[consume_monthly["days_in_a_month"] >= 3]
    consume_by_month = pd.DataFrame({"avg_consume_monthly": consume_monthly.groupby(["stu_id"])["consume_by_month"].mean(),
                                    "median_consume_monthly": consume_monthly.groupby(["stu_id"])["consume_by_month"].median(),
                                    "max_consume_monthly": consume_monthly.groupby(["stu_id"])["consume_by_month"].max(),
                                    "min_consume_monthly": consume_monthly.groupby(["stu_id"])["consume_by_month"].min(),
                                    "avg_count_consume_monthly": consume_monthly.groupby(["stu_id"])["days_in_a_month"].mean()}).reset_index()
    consume_by_month = consume_by_month.astype(int)
    consume_by_month.to_csv("input/processed/consume_by_month.csv", index=False)

    # feature 4: extract summer vacation consume features
    consume_monthly_all = pd.DataFrame({"days_in_a_month":consume_train_test.groupby(["stu_id","year","month"])["day"].count(),
                                "consume_by_month":consume_train_test.groupby(["stu_id","year","month"])["amount"].sum()}).reset_index()
    consume_month_july = consume_monthly_all[consume_monthly_all["month"] == 7]
    consume_month_july = consume_month_july[["stu_id", "consume_by_month", "days_in_a_month"]]
    consume_month_july.columns = ["stu_id", "consume_july", "count_july"]
    consume_month_august = consume_monthly_all[consume_monthly_all["month"] == 8]
    consume_month_august = consume_month_august[["stu_id", "consume_by_month", "days_in_a_month"]]
    consume_month_august.columns = ["stu_id", "consume_august", "count_august"]
    consume_july_august = pd.merge(consume_month_july, consume_month_august, how='left', on='stu_id')
    consume_july_august.to_csv("input/processed/consume_july_august.csv", index=False)


    # feature 5: extract weekend consume features
    consume_train_test["weekday"] = consume_train_test.when.apply(lambda t: t.weekday())
    consume_train_test["weekend_or_not"] = consume_train_test["weekday"].apply(lambda i: 1 if i >= 5 else 0)
    consume_by_weekend = pd.DataFrame({"consume_weekend_count": consume_train_test.groupby(['stu_id', 'weekend_or_not'])['amount'].count(),
                              "consume_weekend_sum": consume_train_test.groupby(['stu_id', 'weekend_or_not'])['amount'].sum(),
                              "consume_weekend_max": consume_train_test.groupby(['stu_id', 'weekend_or_not'])['amount'].max(),
                              "consume_weekend_mean": consume_train_test.groupby(['stu_id', 'weekend_or_not'])['amount'].mean(),
                              "consume_weekend_median": consume_train_test.groupby(['stu_id', 'weekend_or_not'])['amount'].median()}).reset_index()
    consume_by_weekend = consume_by_weekend[consume_by_weekend["weekend_or_not"] == 1]
    consume_by_weekend.drop(["weekend_or_not"], axis=1, inplace=True)
    consume_by_weekend.drop(["consume_weekend_mean"], axis=1, inplace=True)
    consume_by_weekend.to_csv("input/processed/consume_by_weekend.csv", index=False)


    # feature 6: extract some features from dining hall consume data
    consume_of_dining = consume_train_test[consume_train_test["type"] == 0]
    grouped_dining = pd.DataFrame({"consume_count_dining": consume_of_dining.groupby(['where'])['amount'].count(),
                               "consume_mean_dining": consume_of_dining.groupby(['where'])['amount'].mean(),
                               "consume_median_dining": consume_of_dining.groupby(['where'])['amount'].median()}).reset_index()
    sort_by_count = grouped_dining.sort("consume_count_dining", ascending=False)
    sort_by_mean = grouped_dining.sort("consume_mean_dining", ascending=False)
    # pick ten most popular dining halls to extract features
    ten_pop_dining = list(sort_by_count["where"][0:10])
    # combine the sort_by_mean data and the sort_by_median data, pick seven most expensive dining halls to extract features
    seven_exp_dining = list(sort_by_mean["where"][0:7])
    dining_of_interest = ten_pop_dining + seven_exp_dining
    dining_of_interest_df = consume_of_dining[consume_of_dining["where"].isin(dining_of_interest)]
    consume_interest_dining_count = dining_of_interest_df.groupby(['stu_id', 'where'])["amount"].count().unstack().reset_index()
    consume_interest_dining_count.columns = ['stu_id', 'place_1155_count', 'place_118_count', 'place_1551_count', 'place_1683_count', 'place_1985_count', 'place_217_count', 'place_232_count', 'place_247_count', 'place_250_count', 'place_272_count', 'place_275_count', 'place_60_count', 'place_61_count', 'place_69_count', 'place_72_count', 'place_83_count', 'place_841_count']
    consume_interest_dining_max = dining_of_interest_df.groupby(['stu_id', 'where'])["amount"].max().unstack().reset_index()
    consume_interest_dining_max.columns = ['stu_id', 'place_1155_max', 'place_118_max', 'place_1551_max', 'place_1683_max', 'place_1985_max', 'place_217_max', 'place_232_max', 'place_247_max', 'place_250_max', 'place_272_max', 'place_275_max', 'place_60_max', 'place_61_max', 'place_69_max', 'place_72_max', 'place_83_max', 'place_841_max']
    consume_interest_dining_mean = dining_of_interest_df.groupby(['stu_id', 'where'])["amount"].mean().unstack().reset_index()
    consume_interest_dining_mean.columns = ['stu_id', 'place_1155_mean', 'place_118_mean', 'place_1551_mean', 'place_1683_mean', 'place_1985_mean', 'place_217_mean', 'place_232_mean', 'place_247_mean', 'place_250_mean', 'place_272_mean', 'place_275_mean', 'place_60_mean', 'place_61_mean', 'place_69_mean', 'place_72_mean', 'place_83_mean', 'place_841_mean']
    consume_interest_dining_median = dining_of_interest_df.groupby(['stu_id', 'where'])["amount"].median().unstack().reset_index()
    consume_interest_dining_median.columns = ['stu_id', 'place_1155_median', 'place_118_median', 'place_1551_median', 'place_1683_median', 'place_1985_median', 'place_217_median', 'place_232_median', 'place_247_median', 'place_250_median', 'place_272_median', 'place_275_median', 'place_60_median', 'place_61_median', 'place_69_median', 'place_72_median', 'place_83_median', 'place_841_median']
    consume_interest_dining = pd.merge(consume_interest_dining_count, consume_interest_dining_max, how='left', on='stu_id')
    consume_interest_dining = pd.merge(consume_interest_dining, consume_interest_dining_mean, how='left', on='stu_id')
    consume_interest_dining = pd.merge(consume_interest_dining, consume_interest_dining_median, how='left', on='stu_id')
    consume_interest_dining.to_csv("input/processed/consume_interest_dining.csv", index=False)


    # feature 7: extract some features from supermarket consume data
    consume_of_supermarket = consume_train_test[consume_train_test["type"] == 1]
    supermarket_of_interest = ["188", "190", "192", "219", "248"]
    supermarket_of_interest_df = consume_of_supermarket[consume_of_supermarket["where"].isin(supermarket_of_interest)]
    consume_interest_super_count = supermarket_of_interest_df.groupby(['stu_id', 'where'])["amount"].count().unstack().reset_index()
    consume_interest_super_count.columns = ["stu_id", "place_188_count", "place_190_count", "place_192_count", "place_219_count", "place_248_count"]
    consume_interest_super_sum = supermarket_of_interest_df.groupby(['stu_id', 'where'])["amount"].sum().unstack().reset_index()
    consume_interest_super_sum.columns = ["stu_id", "place_188_sum", "place_190_sum", "place_192_sum", "place_219_sum", "place_248_sum"]
    consume_interest_super = pd.merge(consume_interest_super_count, consume_interest_super_sum, how='left', on='stu_id')
    consume_interest_super.to_csv("input/processed/consume_interest_super.csv", index=False)


    # feature 8: consume analysis based on time intervals in a day
    consume_train_test_timesplit = pd.read_csv("input/processed/consume_train_test_timesplit.csv")
    consume_by_hour_count = consume_train_test_timesplit.groupby(["stu_id", "hour"])["amount"].count().unstack().reset_index()
    consume_by_hour_count.columns = ["stu_id", "hour_0_count", "hour_1_count", "hour_2_count", "hour_3_count", "hour_4_count", "hour_5_count", "hour_6_count", "hour_7_count", "hour_8_count", "hour_9_count", "hour_10_count", "hour_11_count", "hour_12_count", "hour_13_count", "hour_14_count", "hour_15_count", "hour_16_count", "hour_17_count", "hour_18_count", "hour_19_count", "hour_20_count", "hour_21_count", "hour_22_count", "hour_23_count"]
    consume_by_hour_sum = consume_train_test_timesplit.groupby(["stu_id", "hour"])["amount"].sum().unstack().reset_index()
    consume_by_hour_sum.columns = ["stu_id", "hour_0_sum", "hour_1_sum", "hour_2_sum", "hour_3_sum", "hour_4_sum", "hour_5_sum", "hour_6_sum", "hour_7_sum", "hour_8_sum", "hour_9_sum", "hour_10_sum", "hour_11_sum", "hour_12_sum", "hour_13_sum", "hour_14_sum", "hour_15_sum", "hour_16_sum", "hour_17_sum", "hour_18_sum", "hour_19_sum", "hour_20_sum", "hour_21_sum", "hour_22_sum", "hour_23_sum"]
    consume_by_hour_median = consume_train_test_timesplit.groupby(["stu_id", "hour"])["amount"].median().unstack().reset_index()
    consume_by_hour_median.columns = ["stu_id", "hour_0_median", "hour_1_median", "hour_2_median", "hour_3_median", "hour_4_median", "hour_5_median", "hour_6_median", "hour_7_median", "hour_8_median", "hour_9_median", "hour_10_median", "hour_11_median", "hour_12_median", "hour_13_median", "hour_14_median", "hour_15_median", "hour_16_median", "hour_17_median", "hour_18_median", "hour_19_median", "hour_20_median", "hour_21_median", "hour_22_median", "hour_23_median"]
    consume_by_hour_max = consume_train_test_timesplit.groupby(["stu_id", "hour"])["amount"].max().unstack().reset_index()
    consume_by_hour_max.columns = ["stu_id", "hour_0_max", "hour_1_max", "hour_2_max", "hour_3_max", "hour_4_max", "hour_5_max", "hour_6_max", "hour_7_max", "hour_8_max", "hour_9_max", "hour_10_max", "hour_11_max", "hour_12_max", "hour_13_max", "hour_14_max", "hour_15_max", "hour_16_max", "hour_17_max", "hour_18_max", "hour_19_max", "hour_20_max", "hour_21_max", "hour_22_max", "hour_23_max"]
    consume_by_hour = pd.merge(consume_by_hour_count, consume_by_hour_sum, how="left", on="stu_id")
    consume_by_hour = pd.merge(consume_by_hour, consume_by_hour_median, how="left", on="stu_id")
    consume_by_hour = pd.merge(consume_by_hour, consume_by_hour_max, how="left", on="stu_id")
    consume_by_hour.to_csv("input/processed/consume_by_hour.csv", index=False)




if __name__ == '__main__':
    print('start extracting features from original data...')
    if not os.path.exists('input/processed'):
        os.makedirs('input/processed')

    start_time = time.time()
    ScoreProcess()
    print('score data processed.')
    DormProcess()
    print('dorm data processed.')
    LibraryProcess()
    print('library data processed.')
    CardProcess()
    print('card data processed.')
    print('finish extracting features. total time: {}'.format(time.time() - start_time))
