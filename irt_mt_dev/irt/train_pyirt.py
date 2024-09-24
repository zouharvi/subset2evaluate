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
args.add_argument("--out", default="computed/pyyirt.json")
args.add_argument("-s", "--seed", type=int, default=None)
args.add_argument("-e", "--epochs", type=int, default=1_000)
args.add_argument("--metric", default="exact_match")
args = args.parse_args()

# data = utils.load_data(normalize=True, binarize=True)
data = utils.load_data_squad(n_items=None, n_systems=None)
systems = list(data[0]["scores"]["human"].keys())

def get_line_class(line, system):
    return line["scores"]["human"][system] == 1.0

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

config = py_irt.config.IrtConfig(model_type='4pl', log_every=500, dropout=0, seed=args.seed)
trainer = py_irt.training.IrtModelTrainer(config=config, data_path=None, dataset=dataset)
trainer.train(epochs=args.epochs, device='cuda')

json.dump({
    "theta": trainer.last_params["ability"],
    "a": trainer.last_params["disc"],
    "b": trainer.last_params["diff"],
    "c": trainer.last_params["lambdas"],
}, open(args.out, "w"))