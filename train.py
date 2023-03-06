import pickle
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
import sklearn
import sys, platform


def data_processing():
    df = pd.read_csv('assets/ames.csv')
    df.columns = df.columns.str.replace(" ", "")

    #print(df.columns)
    # pick some columns, drop the nulls.
    good_cols = [
    "LotFrontage",
    "LotArea",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "SalePrice"
]
    df.dropna(subset=good_cols, inplace=True)

    X, y = df[good_cols[:int(sys.argv[1])]], df['SalePrice']

    return X, y


def lin_reg_model_training(X_train, y_train):
    model = LinearRegression()

    model.fit(X_train, y_train)

    # print out the score and coefficients
    # print(f'The model explains {100*model.score(X,y):.2f}% of the variance' + '\n-----\n' + 'Coefficients:')
    # print(dict(zip(list(X_train.columns), np.round(model.coef_, 4))))

    return model


def save_model(trained_model):
    pickle.dump(trained_model, open('assets/model.p', 'wb'))


def make_prediction(data_array):
    model = pickle.load(open('assets/model.p', 'rb'))
    pred = model.predict(data_array)
    return  pred

if __name__ == '__main__':

    #print(f"OS is {platform.system()}")
    # print(f"Python version {sys.version}")
    # print(f"Pandas version {pd.__version__}")
    # print(f"Numpy version {np.__version__}")
    # print(f"Sklearn version {sklearn.__version__}")


    X, y = data_processing()


    if sys.argv[2] == "lin_reg":
        save_model(lin_reg_model_training(X.iloc[:1500, :], y[:1500]))

    r2 = sklearn.metrics.r2_score(make_prediction(X.iloc[1500:, :]), y[1500:])

    print([platform.system(), sys.version, pd.__version__, np.__version__, sklearn.__version__,
           sys.argv[1], sys.argv[2], r2])
