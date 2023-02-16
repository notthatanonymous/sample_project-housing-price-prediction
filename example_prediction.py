import pickle
import numpy as np


def make_prediction(data_array):
    model = pickle.load(open('assets/model.p', 'rb'))
    pred = model.predict(np.array(data_array).reshape(1, -1))
    print(pred)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    make_prediction([8, 3, 1100, 1500])
