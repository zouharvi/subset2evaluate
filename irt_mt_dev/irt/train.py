import irt_mt_dev.utils as utils
import torch
import torch.utils
import lightning as L
import argparse
from basic import IRTModel


args = argparse.ArgumentParser()
args.add_argument("--metric", default="score")
args.add_argument("--binarize", "--bin", action="store_true")
args = args.parse_args()

data_wmt = utils.load_data(normalize=True, binarize=args.binarize)

systems = list(data_wmt[0]["score"].keys())
model = IRTModel(len(data_wmt), systems)

if args.metric == "score":
    data_loader = [
        ((sent_i, sys_i), sent["score"][sys])
        for sent_i, sent in enumerate(data_wmt)
        for sys_i, sys in enumerate(systems)
    ]
else:
    data_loader = [
        ((sent_i, sys_i), sent["metrics"][sys][args.metric])
        for sent_i, sent in enumerate(data_wmt)
        for sys_i, sys in enumerate(systems)
    ]

data_loader = torch.utils.data.DataLoader(
    data_loader,
    batch_size=len(data_wmt)*len(systems),
    num_workers=24,
    shuffle=True,
    # fully move to GPU
    pin_memory=True,
    # don't kill workers because that's our bottleneck
    persistent_workers=True,
)

trainer = L.Trainer(max_epochs=1000, log_every_n_steps=1)
trainer.fit(model=model, train_dataloaders=data_loader)

suffix = ""
if args.binarize:
    suffix += "_bin"
model.save_irt(f"computed/irt_{args.metric}{suffix}.json")


"""
for METRIC in "MetricX-23-c" "BLEU" "score"; do
for BINARIZE in "" "--binarize"; do
    echo "###" $METRIC $BINARIZE;
    python3 irt_mt_dev/irt/train.py --metric $METRIC $BINARIZE;
done;
done;
"""