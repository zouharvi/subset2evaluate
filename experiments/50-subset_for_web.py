# %%

import subset2evaluate.utils
import subset2evaluate.evaluate
import json
import random

data = subset2evaluate.utils.load_data_wmt(year="wmt24", langs="en-cs")

# %%
sys_abs = subset2evaluate.evaluate.get_model_absolute(data)
sys_abs = list(sys_abs.items())
sys_abs.sort(key=lambda x: x[1], reverse=True)
for i, (name, score) in enumerate(sys_abs):
    print(f"{i+1:2d}. {name:30s} {score:.2f}")

models = ["Unbabel-Tower70B", "ONLINE-W", "IOL_Research", "CUNI-GA", "IKUN-C"]

# %%
data_new = [
    {
        "src": item["src"],
        "scores": {
            model: item["scores"][model]["human"]
            for model in models
        }
    }
    for item in data
    if "http" not in item["src"] and "<div" not in item["src"] and len(item["src"].split()) < 60
]

data_new = random.Random(1).sample(data_new, 100)

with open("../web/src/src/data.json", "w") as f:
    json.dump(data_new, f)