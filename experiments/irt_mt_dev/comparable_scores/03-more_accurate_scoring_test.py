import irt_mt_dev.utils as utils
import collections
import numpy as np

data_old = utils.load_data()

results_all_pred = []
results_all_true = []

for sys_i in range(len(data_old[0]["scores"]["human"].keys())):
    SYSTEMS = list(data_old[0]["scores"]["human"].keys())
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
            for v in [line["scores"][sys]["MetricX-23-c"] for sys in SYSTEMS]
        ])

        data[domain] = [
            (
                # store lookup vector
                np.array([line["scores"][sys]["MetricX-23-c"] >= median for sys in SYSTEMS], dtype=np.float_),
                # store actual scores
                [line["scores"][sys]["MetricX-23-c"] for sys in SYSTEMS],
                # score test passing
                line["scores"][SYSTEM_TEST]["MetricX-23-c"]
            )
            for line in data[domain]
        ]

    def signature_distance(line_a, line_b):
        return np.sum(np.abs(line_a - line_b))

    # match
    data_a = data["social"]
    data_b = data["news"]
    matching = {}
    for line_i, line in enumerate(data_b):
        # TODO: try changing me to other distance thresholds
        good_neighbours = [x[0] for x in enumerate(data_a) if signature_distance(line[0], x[1][0]) <= 1]
        matching[line_i] = good_neighbours

    # predict
    test_prob_correct_all = []
    for line_b_i, lines_a_i in matching.items():
        if lines_a_i:
            pred_scores = []
            for line_a_i in lines_a_i:
                test_score = data_a[line_a_i][2]
                partner_score_i = np.argmin([abs(x-test_score) for x in data_a[line_a_i][1]])
                partner_score_b = data_b[line_b_i][1][partner_score_i]
                pred_scores.append(partner_score_b)
            test_prob_correct_all.append(np.average(pred_scores))

    results_all_pred.append(np.average(test_prob_correct_all))
    results_all_true.append(np.average([line[2] for line in data_b]))

    print("PRED", results_all_pred[-1])
    print("TRUE", results_all_true[-1])
    print()

print("CORR", np.corrcoef(results_all_pred, results_all_true)[0,1])
print("ABS ", np.sum(np.abs(np.array(results_all_pred) - np.array(results_all_true))))


import matplotlib.pyplot as plt
plt.scatter(results_all_true, results_all_pred)
plt.show()