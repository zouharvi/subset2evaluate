import json
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import irt_mt_dev.utils as utils
import numpy as np

data_wmt = utils.load_data(normalize=True, binarize=False)
systems = list(data_wmt[0]["score"].keys())
# data_loader = [
#     ((sent_i, sys_i), sent["score"][sys])
#     for sent_i, sent in enumerate(data_wmt)
#     for sys_i, sys in enumerate(systems)
# ]
data_loader = [
    ((sent_i, sys_i), sent["metrics"][sys]["MetricX-23-c"])
    for sent_i, sent in enumerate(data_wmt)
    for sys_i, sys in enumerate(systems)
]

encoder_item = OneHotEncoder().fit([[x[0][0]] for x in data_loader])
encoder_system = OneHotEncoder().fit([[x[0][1]] for x in data_loader])
data_train, data_test = train_test_split(data_loader, random_state=0, train_size=0.9)

X_train = [np.concatenate((encoder_item.transform([[x[0][0]]]).toarray().flatten(), encoder_system.transform([[x[0][1]]]).toarray().flatten())) for x in data_train]
Y_train = [x[1] for x in data_train]
X_test = [np.concatenate((encoder_item.transform([[x[0][0]]]).toarray().flatten(), encoder_system.transform([[x[0][1]]]).toarray().flatten())) for x in data_test]
Y_test = [x[1] for x in data_test]

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print(f"MAE: {np.average([abs(y_pred - y_true) for y_pred, y_true in zip(Y_pred, Y_test)]):.3f}")
print(f"Corr: {np.corrcoef(Y_test, Y_pred)[0,1]:.3f}")