"""Train module"""

import pandas as pd
import xgboost as xgb
import numpy as np
import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

def use_cv() -> GridSearchCV():
    """
    Applies Cross Validation to XGBoost Regressor
    :return: Model ready to fit
    """
    param_test1 = {
        'learning_rate': [0.35, 0.4, 0.5],
        'max_depth': [10],
        'min_child_weight': [9],
        'colsample_bytree': [0.8, 0.9, 1]
    }
    xg_reg = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3,
                                                     alpha=10, n_estimators=10),
                          param_grid=param_test1, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1)

    return xg_reg

def use_cv_random_forest() -> GridSearchCV():
    """
    Applies Cross Validation to Random Forest Regressor
    :return: Model ready to fit
    """
    param_grid = {
        'bootstrap': [True],
        'max_depth': [90, 100],
        'max_features': [2, 3],
        'min_samples_leaf': [4, 5],
        'min_samples_split': [10, 12],
        'n_estimators': [200, 300]
    }
    # Create a based model1
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    regr = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)

    return regr

def evaluate_model(y_test: pd.DataFrame, pred: pd.DataFrame) -> int:
    """
    Measure and show all the valueable metrics
    :param y_test: Ground Truth
    :param pred: Predictions
    :return: 1 if success
    """
    rmsle = round(np.sqrt(mean_squared_log_error(y_test, pred)), 10)
    rmse = round(np.sqrt(mean_squared_error(y_test, pred)), 2)
    mae = round(mean_absolute_error(y_test, pred), 2)
    mape = round(100*mean_absolute_percentage_error(y_test, pred), 2)
    print(f"RMSLE: {rmsle}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape} %")

    return 1

def train_model_xgboost(df: pd.DataFrame(), cols_to_scale: np.array(), cv: bool = True) -> int:
    '''
    # https://www.projectpro.io/recipes/perform-xgboost-algorithm-with-sklearn
    :param df: DataFrame to train the model
    :param cols_to_scale: Columns that requires to scale
    :param cv: Cross Validation control execution
    :return: 1 if success
    '''
    print("Training XGBoost --------------")
    X, y = df.loc[:, df.columns != 'cost'], df.loc[:, df.columns == 'cost']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    ss = StandardScaler()
    x_train[cols_to_scale] = ss.fit_transform(x_train[cols_to_scale])
    x_test[cols_to_scale] = ss.transform(x_test[cols_to_scale])

    if cv:
        xg_reg = use_cv()
        xg_reg.fit(x_train, y_train)
        print('Best Params: ')
        print(xg_reg.best_params_)

    else:
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5,
                                  alpha=10, n_estimators=10)
        xg_reg.fit(x_train, y_train)
    pred = xg_reg.predict(x_test)

    evaluate_model(y_test, pred)

    return 1

def train_model_random_forest(df: pd.DataFrame(), cols_to_scale: np.array(), cv: bool = True) -> int:
    '''
    Train the Random Forest
    :param df: DataFrame to train the model
    :param cols_to_scale: Columns that requires to scale
    :param cv: Cross Validation control execution
    :return: 1 if success
    '''
    print("Training Random Forest --------------")
    X, y = df.loc[:, df.columns != 'cost'], df.loc[:, df.columns == 'cost']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    ss = StandardScaler()
    x_train[cols_to_scale] = ss.fit_transform(x_train[cols_to_scale])
    x_test[cols_to_scale] = ss.transform(x_test[cols_to_scale])

    if cv:
        regr = use_cv_random_forest()
        regr.fit(x_train, y_train)
        print('Best Params: ')
        print(regr.best_params_)
    else:
        regr = RandomForestRegressor(max_depth=2, random_state=0)
        regr.fit(x_train, y_train)
    pred = regr.predict(x_test)

    evaluate_model(y_test, pred)

    return 1

def train_model_lgbm(df: pd.DataFrame) -> int:
    '''
    # https://www.datatechnotes.com/2022/03/lightgbm-regression-example-in-python.html

    :param df: dataframe to train the model
    :return: 1 if run worked fine
    '''
    print("Training lgbm --------------")
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    params = {
        'task': 'train',
        'boosting': 'gbdt',
        'objective': 'regression',
        'num_leaves': 10,
        'learnnig_rage': 0.05,
        'metric': {'l2', 'l1'},
        'verbose': -1
    }
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    model = lgb.train(params,
                      train_set=lgb_train,
                      valid_sets=lgb_eval,
                      early_stopping_rounds=30)

    pred = model.predict(x_test)

    evaluate_model(y_test, pred)

    return 1


