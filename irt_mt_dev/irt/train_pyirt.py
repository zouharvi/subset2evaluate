import py_irt
import py_irt.config
import py_irt.dataset
import py_irt.io
import py_irt.training
import irt_mt_dev.utils as utils
import argparse
import numpy as np
import json


args = argparse.ArgumentParser()
args.add_argument("--metric", default="score")
args = args.parse_args()

data = utils.load_data(normalize=True, binarize=True)
systems = list(data[0]["metrics"].keys())

def get_line_score(line, system):
    if args.metric == "score":
        return line["score"][system]
    else:
        return line["metrics"][system][args.metric]

def get_line_class(line, system):
    return bool(get_line_score(line, system))

py_irt.io.write_jsonlines(
    "/tmp/irt_dataset.jsonl",
    [
        {
            "subject_id": system,
            "responses": {f"item_{line_i}": get_line_class(line, system) for line_i, line in enumerate(data)}
        }
        for system in systems
    ]
)

dataset = py_irt.dataset.Dataset.from_jsonlines("/tmp/irt_dataset.jsonl")

config = py_irt.config.IrtConfig(model_type='2pl', log_every=500, dropout=0)
trainer = py_irt.training.IrtModelTrainer(config=config, data_path=None, dataset=dataset)
trainer.train(epochs=5_000, device='cuda')

json.dump({
    "theta": trainer.last_params["ability"],
    "a": trainer.last_params["disc"],
    "b": trainer.last_params["diff"],
}, open(f"computed/2pl_{args.metric}.json", "w"))