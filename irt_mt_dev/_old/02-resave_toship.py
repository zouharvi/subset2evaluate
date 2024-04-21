import pickle
import collections
from pympler.asizeof import asizeof
import tqdm

data_raw = pickle.load(open("data/toship21.pkl", "rb"))
print("Original object is", asizeof(data_raw) // (1024**2), "MB")

# map lines to systems 
data_new = collections.defaultdict(dict)
for doc_name, doc in tqdm.tqdm(list(data_raw.items())):
    for sys in doc.values():
        sys_name = sys["automatic_metrics"]["System"]
        for _line_i, line in sys["hum_annotations"].iterrows():
            # TODO: do we care about source and target langauges?
            data_new[line["Segment"] + " | " + line["Reference"]][sys_name] = {
                "score": line["Score"],
                "metrics": {k: v for k, v in line.to_dict().items() if k.startswith("metric_")},
            }

print("New object is", asizeof(data_new) // (1024**2), "MB")

data_new = [
    # drop line name, we don't need it
    line for line in data_new.values()
    # take only lines that have at least two systems
    if len(line) > 1
]

print("Filtered object is", asizeof(data_new) // (1024**2), "MB")

with open("data/toship21.simple.pkl", "wb") as f:
    pickle.dump(data_new, f)