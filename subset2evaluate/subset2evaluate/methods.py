import numpy as np
import irt_mt_dev.utils as utils
from functools import partial


def random(data, **kwargs):
    import random
    random.Random(0).shuffle(data)
    return data


def metric_avg(data, **kwargs):
    data.sort(key=lambda item: np.average(
        [sys_v[kwargs["metric"]] for sys_v in item["scores"].values()]
    ))
    return data


def metric_var(data, **kwargs):
    data.sort(key=lambda item: np.var(
        [sys_v[kwargs["metric"]] for sys_v in item["scores"].values()]
    ), reverse=True)
    return data


def metric_consistency(data, **kwargs):
    metric_scores = utils.get_sys_absolute(data, metric=kwargs["metric"])
    rank_correlation = {}
    sys_names = list(metric_scores.keys())
    for example in data:
        consistency = 0
        total = 0
        for i in range(len(sys_names)):
            for j in range(i+1, len(sys_names)):
                if (metric_scores[sys_names[i]] - metric_scores[sys_names[j]]) * (example['scores'][sys_names[i]][kwargs["metric"]] - example['scores'][sys_names[j]][kwargs["metric"]]) > 0:
                    consistency += 1
                else:
                    consistency -= 1
                total += 1
        rank_correlation[example['i']] = consistency / total
    data.sort(key=lambda item: rank_correlation[item['i']], reverse=True)
    return data


def irt(data, **kwargs):
    import torch
    import torch.utils
    import lightning as L
    import irt_mt_dev.utils as utils
    # turn off pesky pytorch logs
    import logging
    logging.disable(logging.INFO)

    systems = list(data[0]["scores"].keys())

    if kwargs["model"] == "scalar":
        from irt_mt_dev.irt.scalar import IRTModelScalar
        model = IRTModelScalar(data, systems, **kwargs)
    elif kwargs["model"] == "tfidf":
        from irt_mt_dev.irt.tfidf import IRTModelTFIDF
        model = IRTModelTFIDF(data, systems, **kwargs)
    elif kwargs["model"] == "embd":
        from irt_mt_dev.irt.embd import IRTModelEmbd
        model = IRTModelEmbd(data, systems, **kwargs)
    else:
        raise Exception("Model not defined")

    data_loader = [
        ((sent_i, sys_i), sent["scores"][sys][kwargs["metric"]])
        for sent_i, sent in enumerate(data)
        for sys_i, sys in enumerate(systems)
    ]

    # TODO: in the future run first training with dev set to find out the best epoch count
    # and then run again on full data with that epoch count
    data_train = data_loader

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
        max_epochs=500 if kwargs["model"] != "embd" else 2000,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(
        model=model,
        train_dataloaders=data_train,
    )


    data_irt = model.pack_irt_params()
    items_joint = list(zip(data, data_irt["items"]))

    items_joint.sort(key=lambda x: model.fn_utility(x[1], data_irt["systems"]), reverse=True)

    return [x[0] for x in items_joint]

METHODS = {
    "random": random,
    "avg": metric_avg,
    "var": metric_var,
    "irt_diff": partial(irt, fn_utility="diff"),
    "irt_disc": partial(irt, fn_utility="disc"),
    "irt_feas": partial(irt, fn_utility="feas"),
    "irt_fic": partial(irt, fn_utility="fisher_information_content"),
}