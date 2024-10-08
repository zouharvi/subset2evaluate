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
args.add_argument("-d", "--data", choices=["squad", "wmt"], default="wmt")
args.add_argument("-e", "--epochs", type=int, default=1_000)
args.add_argument("--metric", default="exact_match")
args.add_argument("--all-params", action="store_true")
args = args.parse_args()

if args.data == "wmt":
    data = utils.load_data(normalize=True)
elif args.data == "squad":
    data = utils.load_data_squad(n_items=None, n_systems=None)

systems = list(data[0]["scores"].keys())

median = np.median([
    line["scores"][system][args.metric]
    for line in data
    for system in systems
])
print("Median", f"{median:.2f}")

py_irt.io.write_jsonlines(
    "/tmp/irt_dataset.jsonl",
    [
        {
            "subject_id": system,
            "responses": {f"item_{line_i}": bool(line["scores"][system][args.metric] >= median) for line_i, line in enumerate(data)}
        }
        for system in systems
    ]
)

dataset = py_irt.dataset.Dataset.from_jsonlines("/tmp/irt_dataset.jsonl")

config = py_irt.config.IrtConfig(
    model_type='4pl',
    log_every=100,
    # dropout=0,
    seed=args.seed,
    # TODO: see what this does
    deterministic=True,
)
trainer = py_irt.training.IrtModelTrainer(
    config=config,
    data_path=None,
    dataset=dataset,
)
trainer.train(epochs=args.epochs, device='cuda')

json.dump([
    {
        "systems": {sys: sys_v for sys, sys_v in zip(systems, params["ability"])},
        "items": [
            {"disc": disc, "diff": diff, "feas": feas}
            for disc, diff, feas in zip(
                params["disc"],
                params["diff"],
                params["lambdas"],
            )
        ]
    }
    for params
    in (trainer.all_params if args.all_params else [trainer.last_params])
], open(args.out, "w"))