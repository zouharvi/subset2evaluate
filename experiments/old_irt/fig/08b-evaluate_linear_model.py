from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import subset2evaluate.utils as utils
import numpy as np

data_wmt = utils.load_data_wmt(normalize=True, binarize=False)
systems = list(data_wmt[0]["scores"].keys())
data_loader = [
    ((sent_i, sys_i), sent["scores"][sys]["MetricX-23-c"])
    for sent_i, sent in enumerate(data_wmt)
    for sys_i, sys in enumerate(systems)
]

encoder_item = OneHotEncoder().fit([[x[0][0]] for x in data_loader])
encoder_system = OneHotEncoder().fit([[x[0][1]] for x in data_loader])
data_train, data_test = train_test_split(data_loader, random_state=0, train_size=0.9)


def evaluate_lm_prediction(use_sample, use_system):
    def encode(x):
        out = []
        if use_sample:
            out.append(encoder_item.transform([[x[0][0]]]).toarray().flatten())
        if use_system:
            out.append(encoder_system.transform([[x[0][1]]]).toarray().flatten())

        if out:
            return np.concatenate(out)
        else:
            return np.array([0])

    X_train = [encode(x) for x in data_train]
    Y_train = [x[1] for x in data_train]
    X_test = [encode(x) for x in data_test]
    Y_test = [x[1] for x in data_test]

    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    Y_pred = np.clip(Y_pred, 0, 1)
    print()
    print("Features: ")
    if use_sample:
        print("- samples")
    if use_system:
        print("- systems")
    if not use_sample and not use_system:
        print("- none")
    print(f"MAE:  {np.average([abs(y_pred - y_true) for y_pred, y_true in zip(Y_pred, Y_test)]):.3f}")
    print(f"Corr: {np.corrcoef(Y_test, Y_pred)[0, 1]:.3f}")


evaluate_lm_prediction(use_sample=True, use_system=True)
evaluate_lm_prediction(use_sample=True, use_system=False)
evaluate_lm_prediction(use_sample=False, use_system=True)
evaluate_lm_prediction(use_sample=False, use_system=False)
