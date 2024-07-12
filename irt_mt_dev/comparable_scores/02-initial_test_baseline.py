import irt_mt_dev.utils as utils
import collections
import numpy as np

data_old = utils.load_data()

results_all_pred = []
results_all_true = []

for sys_i in range(len(data_old[0]["score"].keys())):
    SYSTEMS = list(data_old[0]["score"].keys())
    SYSTEM_TEST = SYSTEMS.pop(sys_i)

    data = collections.defaultdict(list)

    # split into domains
    for line in data_old:
        data[line["domain"]].append(line)


    # binarize and vectorize model results
    for domain in data.keys():
        median = np.median([
            v
            for line in data[domain]
            for v in [line["metrics"][sys]["MetricX-23-c"] for sys in SYSTEMS]
        ])

        data[domain] = [
            (
                np.array([line["metrics"][sys]["MetricX-23-c"] >= median for sys in SYSTEMS], dtype=np.float_),
                line["metrics"][SYSTEM_TEST]["MetricX-23-c"] >= median
            )
            for line in data[domain]
        ]

    def signature_distance(line_a, line_b):
        return np.sum(np.abs(line_a - line_b))

    # match
    data_a = data["social"]
    data_b = data["news"]

    # predict
    test_prob_correct_all = []
    for line_b in data_b:
        test_prob_correct_all.append(np.average(line_b[0]))

    results_all_pred.append(np.average(test_prob_correct_all))
    results_all_true.append(np.average([line[1] for line in data_b]))

    print("PRED", results_all_pred[-1])
    print("TRUE", results_all_true[-1])
    print()

print(np.corrcoef(results_all_pred, results_all_true)[0,1])