import irt_mt_dev.utils as utils
import json
import numpy as np

data_wmt = utils.load_data(normalize=True)

systems = list(data_wmt[0]["score"].keys())
data_out = [
    {
        "subject_id": sys,
        "responses": {
            f"{line_i}": (line["score"][sys] >= 0.9)*1.0
            for line_i, line in enumerate(data_wmt)
        }
    }
    for sys in systems
]


with open("computed/trainpyirt.jsonl", "w") as f:
    f.write("\n".join([json.dumps(line) for line in data_out]))