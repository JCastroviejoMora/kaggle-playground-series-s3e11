# https://www.kaggle.com/competitions/playground-series-s3e11/code?competitionId=47790

from preprocess.processing_data import *
from training.modeling import *
import time
#import config

target = 'cost'
cols_to_discard = ['id', 'gross_weight', 'recyclable_package', 'low_fat', 'units_per_case']
one_hot = False
cv = True

def measure_running_time(st):
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', str(round((elapsed_time/60),2)), 'minutes')
    print("End!")
    return 1

def main():
    st = time.time()
    df = get_data('train.csv')
    # describe_data_scatter(df)
    # make_correlation(df)

    check_quality(df)

    discard_columns(df, cols_to_discard)

    if one_hot:
        to_encode = ['total_children', 'num_children_at_home',
                     'avg_cars_at home(approx).1', 'store_sqft']
        apply_one_hot_encoder(df, to_encode)
    else:
        to_encode = []

    to_encode.append(target)
    cols_to_scale = df.columns.difference(to_encode)

    train_model_xgboost(df, cols_to_scale, cv)

    train_model_random_forest(df, cols_to_scale, cv)

    train_model_lgbm(df)

    measure_running_time(st)

if __name__ == "__main__":
    main()