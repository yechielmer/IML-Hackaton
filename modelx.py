import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from preprocessing import Preprocessing


def error(y, y_hat):
    return {
        'r2': r2_score(y, y_hat),
        'rmse': np.sqrt(mean_squared_error(y, y_hat))
    }


def get_X_y(df):
    return df.drop(['revenue', 'vote_average'], axis=1), df[['revenue', 'vote_average']]


MODEL_NAMES = {'vote_only', 'revenue_only', 'vote', 'revenue'}

# MODEL = XGBRegressor
# SETTINGS = {'colsample_bytree': 0.6, 'gamma': 0.7,'max_depth': 50, 'min_child_weight': 5, 'subsample': 0.8, 'objective': 'reg:squarederror'}

MODEL = RandomForestRegressor
SETTINGS = {'max_depth': 10, 'n_estimators': 300, 'max_features': 150}
# SETTINGS = {'max_depth': 20, 'n_estimators': 100}


def get_models():
    return {
        # 'vote': MODEL(max_depth=MAX_DEPTH, n_estimators=ESTIMATORS, max_features=MAX_FEATURES),
        'vote': MODEL(**SETTINGS),
        # 'revenue': MODEL(max_depth=MAX_DEPTH, n_estimators=ESTIMATORS, max_features=MAX_FEATURES),
        'revenue': MODEL(**SETTINGS),
    }


class Model:
    def __init__(self):
        self.model_zero_budget = get_models()
        self.model = get_models()
        self.single_model = get_models()

    def fit(self, df):
        X_zero, y_zero = get_X_y(df[df['budget'] == 0])
        X_budget, y_budget = get_X_y(df[df['budget'] > 0])

        if len(X_zero) > 0:
            self.model_zero_budget['vote'].fit(X_zero, y_zero['vote_average'])
            self.model_zero_budget['revenue'].fit(X_zero, y_zero['revenue'])

        if len(X_budget) > 0:
            self.model['vote'].fit(X_budget, y_budget['vote_average'])
            self.model['revenue'].fit(X_budget, y_budget['revenue'])

    def save(self):
        pass

    def predict(self, X):
        # get numbering to re-organize at the end
        X['id'] = np.arange(len(X))

        X_zero = X[X['budget'] == 0]
        X_budget = X[X['budget'] > 0]

        if len(X_zero) > 0:
            vote_zero = self.model_zero_budget['vote'].predict(X_zero.drop('id', axis=1))
            revenue_zero = self.model_zero_budget['revenue'].predict(X_zero.drop('id', axis=1))
            vote_zero = pd.DataFrame({'id': X_zero['id'], 'vote_average': vote_zero})
            revenue_zero = pd.DataFrame({'id': X_zero['id'], 'revenue': revenue_zero})
        else:
            vote_zero = pd.DataFrame({'id': [], 'vote_average': []})
            revenue_zero = pd.DataFrame({'id': [], 'revenue': []})

        if len(X_budget) > 0:
            vote_budget = self.model['vote'].predict(X_budget.drop('id', axis=1))
            revenue_budget = self.model['revenue'].predict(X_budget.drop('id', axis=1))
            vote_budget = pd.DataFrame({'id': X_budget['id'], 'vote_average': vote_budget})
            revenue_budget = pd.DataFrame({'id': X_budget['id'], 'revenue': revenue_budget})
        else:
            vote_budget = pd.DataFrame({'id': [], 'vote_average': []})
            revenue_budget = pd.DataFrame({'id': [], 'revenue': []})

        vote_y = pd.concat([vote_zero, vote_budget])
        revenue_y = pd.concat([revenue_zero, revenue_budget])
        vote_y.loc[vote_y['id'].isin(X[X['is_released'] == 0]['id'])]['vote_average'] = 0
        revenue_y.loc[revenue_y['id'].isin(X[X['is_released'] == 0]['id'])]['revenue'] = 0

        vote_y.sort_values(by=['id'], inplace=True)
        revenue_y.sort_values(by=['id'], inplace=True)

        return revenue_y['revenue'].tolist(), vote_y['vote_average'].round(decimals=1).tolist()

    def error(self, X, y):
        y_hat = self.predict(X)
        return error(y, y_hat)


def generate_data(path):
    return Preprocessing(path).df


def train(path):
    tr = generate_data(path)

    model = Model()

    model.fit(tr)
    return model


def main():
    model = train('movies_dataset.csv')
    print(model)


if __name__ == '__main__':
    main()

