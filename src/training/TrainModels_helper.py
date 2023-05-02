

def get_grid(model, input_size):

    if model == 'Random Forest':
        return {
            'bootstrap': [True],
            'max_depth': [90, 100],
            'max_features': [2, 3],
            'min_samples_leaf': [4, 5],
            'min_samples_split': [10, 12],
            'n_estimators': [200, 300]
        }
    elif model == 'lgbm':
        return {}
    elif model == 'XGBoost':
        return {
            'learning_rate': [0.35, 0.4, 0.5],
            'max_depth': [10],
            'min_child_weight': [9],
            'colsample_bytree': [0.8, 0.9, 1]
        }
    else:
        return {}