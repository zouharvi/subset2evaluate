import json
from sklearn.model_selection import train_test_split
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

data_train, data_test = train_test_split(data_loader, random_state=0, train_size=0.9)

# data_irt = json.load(open("computed/irt_score_0.json"))
data_irt = json.load(open("computed/irt_MetricX-23-c_2k.json"))

def predict_score(system_theta, item_a, item_b):
    return 1 / (1 + np.exp(-item_a * (system_theta - item_b)))

error_model = []
error_constant = []
data_test_avg = np.average([x[1] for x in data_test])
test_pred = []

for (item_i, system_i), score_true in data_test:
    score_pred = predict_score(data_irt["systems"][systems[system_i]], data_irt["items"][item_i]["a"], data_irt["items"][item_i]["b"])
    test_pred.append(score_pred)
    error_model.append(abs(score_pred - score_true))
    error_constant.append(abs(data_test_avg - score_true))

print(f"MAE: {np.average(error_model):.3f}")
print(f"MAE constant: {np.average(error_constant):.3f}")
print(f"Corr: {np.corrcoef([x[1] for x in data_test], test_pred)[0,1]:.3f}")
