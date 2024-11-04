import numpy as np
from functools import partial

def random(data, args):
    import random
    random.Random(0).shuffle(data)
    return data

def metric_avg(data, args):
    data.sort(key=lambda item: np.average(
        [sys_v[args.metric] for sys_v in item["scores"].values()]
    ))
    return data

def metric_var(data, args):
    data.sort(key=lambda item: np.var(
        [sys_v[args.metric] for sys_v in item["scores"].values()]
    ), reverse=True)
    return data

def irt(data, args, selection):
    import torch
    import torch.utils
    import lightning as L
    import irt_mt_dev.utils as utils

    systems = list(data[0]["scores"].keys())

    if args.model == "scalar":
        from irt_mt_dev.irt.scalar import IRTModelScalar
        model = IRTModelScalar(data, systems)
    elif args.model == "tfidf":
        from irt_mt_dev.irt.tfidf import IRTModelTFIDF
        model = IRTModelTFIDF(data, systems)
    elif args.model == "embd":
        from irt_mt_dev.irt.embd import IRTModelEmbd
        model = IRTModelEmbd(data, systems)

    data_loader = [
        ((sent_i, sys_i), sent["scores"][sys][args.metric])
        for sent_i, sent in enumerate(data)
        for sys_i, sys in enumerate(systems)
    ]

    # TODO: in the future run first training with dev set to find out the best epoch count
    # and then run again on full data with that epoch count
    data_train= data_loader

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

    trainer = L.Trainer(
        # for scalar and tfidf, 500 is best
        # for embd, more is needed (2k?)
        max_epochs=2000,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(
        model=model,
        train_dataloaders=data_train,
    )


    data_irt = model.pack_irt_params()
    items_joint = list(zip(data, data_irt["items"]))

    def fn_disc(item):
        return item["disc"]
    
    def fn_information_content(item):
        information = 0
        for theta in data_irt["systems"].values():
            prob = utils.pred_irt(
                theta,
                item
            )
            information += prob*(1-prob)*(item["disc"]**2)
        return information

    if selection == "diff":
        fn_utility = fn_disc
    elif selection == "information_content":
        fn_utility = fn_information_content

    items_joint.sort(key=lambda x: fn_utility(x[1]), reverse=True)

    return [x[0] for x in items_joint]

METHODS = {
    "random": random,
    "avg": metric_avg,
    "var": metric_var,
    "irt_diff": partial(irt, selection="diff"),
    "irt_ic": partial(irt, selection="information_content"),
}