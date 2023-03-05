import pickle
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
import sklearn
import sys, platform

print(f"OS is {platform.system()}")
print(f"Python version {sys.version}")
print(f"Pandas version {pd.__version__}")
print(f"Numpy version {np.__version__}")
print(f"Sklearn version {sklearn.__version__}")


def data_processing():
    df = pd.read_csv('assets/ames.csv')

    print(df.columns)
    # pick some columns, drop the nulls.
    good_cols = ['Overall Qual', 'Full Bath', 'Garage Area', 'Lot Area']
    df.dropna(subset=good_cols, inplace=True)

    X, y = df[good_cols], df['SalePrice']

    return X, y


def model_training(X_train, y_train):
    model = LinearRegression()

    model.fit(X_train, y_train)

    # print out the score and coefficients
    print(f'The model explains {100*model.score(X,y):.2f}% of the variance' + '\n-----\n' + 'Coefficients:')
    print(dict(zip(list(X_train.columns), np.round(model.coef_, 4))))

    return model


def save_model(trained_model):
    pickle.dump(trained_model, open('assets/model.p', 'wb'))


if __name__ == '__main__':
    X, y = data_processing()

    save_model(model_training(X, y))
