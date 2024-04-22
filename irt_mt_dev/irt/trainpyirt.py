import irt_mt_dev.utils as utils
import json

data_wmt = utils.load_data()

systems = list(data_wmt[0]["score"].keys())
data_out = [
    {
        "subject_id": sys,
        "responses": {line_i: (line["metrics"][sys]["MetricX-23-c"]>-1.8)*1.0 for line_i, line in enumerate(data_wmt)}
    }
    for sys in systems
]


with open("computed/trainpyirt.jsonl", "w") as f:
    f.write("\n".join([json.dumps(line) for line in data_out]))