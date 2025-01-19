import json
from sklearn.model_selection import train_test_split
import subset2evaluate.utils as utils
import numpy as np

data_wmt = utils.load_data_wmt(normalize=True, binarize=False)
models = list(data_wmt[0]["scores"].keys())
data_loader = [
    ((sent_i, model_i), sent["scores"][model]["MetricX-23-c"])
    for sent_i, sent in enumerate(data_wmt)
    for model_i, model in enumerate(models)
]

data_train, data_test = train_test_split(data_loader, random_state=0, train_size=0.9)

data_irt = json.load(open("computed/irt_wmt_4pl_s0_our.json"))[5]

error_model = []
test_pred = []

for (item_i, model_i), score_true in data_test:
    score_pred = utils.pred_irt(
        data_irt["models"][models[model_i]],
        data_irt["items"][item_i]
    )
    test_pred.append(score_pred)
    error_model.append(abs(score_pred - score_true))

print(f"MAE:  {np.average(error_model):.3f}")
print(f"Corr: {np.corrcoef([x[1] for x in data_test], test_pred)[0, 1]:.3f}")
