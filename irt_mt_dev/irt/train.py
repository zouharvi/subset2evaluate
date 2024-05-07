import irt_mt_dev.utils as utils
import torch
import torch.utils
import lightning as L
import argparse
from basic import IRTModel
import random
from sklearn.model_selection import train_test_split

args = argparse.ArgumentParser()
args.add_argument("--metric", default="score")
args.add_argument("--binarize", "--bin", action="store_true")
args.add_argument("--train-size", type=float, default=1.0)
args.add_argument("--no-save", action="store_true")
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


# subsample training data
assert args.train_size > 0.0 and args.train_size <= 1.0
if args.train_size == 1.0:
    data_train= data_loader
    data_test = []
else:
    data_train, data_test = train_test_split(data_loader, train_size=args.train_size)

data_train = torch.utils.data.DataLoader(
    data_train,
    batch_size=len(data_train),
    num_workers=24,
    shuffle=True,
    # fully move to GPU
    pin_memory=True,
    # don't kill workers because that's our bottleneck
    persistent_workers=True,
)
data_test = torch.utils.data.DataLoader(
    data_test,
    batch_size=len(data_test),
    num_workers=24,
    shuffle=False,
    # fully move to GPU
    pin_memory=True,
    # don't kill workers because that's our bottleneck
    persistent_workers=True,
)

trainer = L.Trainer(
    max_epochs=1000,
    log_every_n_steps=1,
    check_val_every_n_epoch=500,
    enable_checkpointing=not args.no_save,
    logger=not args.no_save,
)
trainer.fit(
    model=model,
    train_dataloaders=data_train,
    val_dataloaders=data_test,
)

suffix = ""
if args.binarize:
    suffix += "_bin"
if args.train_size != 1.0:
    suffix += f"_sub{args.train_size:.1f}".replace("0.", "0")

if not args.no_save:
    model.save_irt(f"computed/irt_{args.metric}{suffix}.json")