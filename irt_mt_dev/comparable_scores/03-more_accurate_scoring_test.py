import irt_mt_dev.utils as utils
import collections
import numpy as np

data_old = utils.load_data()
SYSTEMS = list(data_old[0]["score"].keys())
SYSTEM_TEST = SYSTEMS.pop(2)

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
            # store lookup vector
            np.array([line["metrics"][sys]["MetricX-23-c"] >= median for sys in SYSTEMS], dtype=np.float_),
            # store actual scores
            [line["metrics"][sys]["MetricX-23-c"] for sys in SYSTEMS],
            # score test passing
            line["metrics"][SYSTEM_TEST]["MetricX-23-c"]
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
    good_neighbours = [x[0] for x in enumerate(data_a) if signature_distance(line[0], x[1][0]) <= 0]
    matching[line_i] = good_neighbours

def get_average_score(line_sig, line_scores, success: int):
    vals = [line_score for line_sig, line_score in zip(line_sig, line_scores) if line_sig == success]
    if vals:
        return np.average(vals)
    else:
        return None
    
def safe_average(vals):
    vals = [val for val in vals if val is not None]
    if vals:
        return np.average(vals)
    else:
        return None

# predict
test_prob_correct_all = []
for line_b_i, lines_a_i in matching.items():
    if lines_a_i:
        test_prob_correct = np.average([data_a[line_a_i][1] for line_a_i in lines_a_i])
        score_when_0 = safe_average([get_average_score(data_a[line_a_i][0], data_a[line_a_i][1], success=0) for line_a_i in lines_a_i])
        score_when_1 = safe_average([get_average_score(data_a[line_a_i][0], data_a[line_a_i][1], success=1) for line_a_i in lines_a_i]) or score_when_0
        score_when_0 = score_when_0 or score_when_1
        test_prob_correct_all.append(test_prob_correct * score_when_1 + (1 - test_prob_correct) * score_when_0)

print("PRED", np.average(test_prob_correct_all))
print("TRUE", np.average([line[2] for line in data_b]))