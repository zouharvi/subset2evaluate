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
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping
    from lightning.pytorch.callbacks.callback import Callback
    from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
    from irt_mt_dev.irt.scalar import IRTModelScalar
    from irt_mt_dev.irt.tfidf import IRTModelTFIDF
    from irt_mt_dev.irt.embd import IRTModelEmbd
    # turn off pesky pytorch logs
    import logging
    logging.disable(logging.INFO)

    systems = list(data[0]["scores"].keys())

    ModelClass = {
        "scalar": IRTModelScalar,
        "tfidf": IRTModelTFIDF,
        "embd": IRTModelEmbd,
    }[kwargs["model"]]
    model = ModelClass(data, systems, data_old=data, **kwargs)

    data_flat = [
        ((sent_i, sys_i), sent["scores"][sys][kwargs["metric"]])
        for sent_i, sent in enumerate(data)
        for sys_i, sys in enumerate(systems)
    ]

    # TODO: in the future run first training with dev set to find out the best epoch count
    # and then run again on full data with that epoch count

    # use all data for both training and validation for now
    data_train = torch.utils.data.DataLoader(
        data_flat,
        batch_size=len(data_flat),
        num_workers=24,
        shuffle=True,
        # fully move to GPU
        pin_memory=True,
        # don't kill workers because that's our bottleneck
        persistent_workers=True,
    )
    data_val = torch.utils.data.DataLoader(
        data_flat,
        batch_size=len(data_flat),
        num_workers=24,
        shuffle=False,
        # fully move to GPU
        pin_memory=True,
        # don't kill workers because that's our bottleneck
        persistent_workers=True,
    )

    # tiny handler to propagate ctrl-C instead of just stopping the training
    class PropagateExitCallback(Callback):
        def on_exception(self, trainer, pl_module, exception):
            exit()

        
    trainer = L.Trainer(
        max_epochs=2000,
        check_val_every_n_epoch=10,
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        # logger=False,
        callbacks=[
            EarlyStopping(monitor="cluster_count_metric", patience=50, verbose=True, mode="max"),
            ModelCheckpoint(filename='best_model', monitor='cluster_count_metric', mode='max', save_top_k=1),
            PropagateExitCallback(),
        ]
    )

    trainer.fit(
        model=model,
        train_dataloaders=data_train,
        val_dataloaders=data_val,
    )

    # reload best model
    model.eval()
    model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path, weights_only=True)["state_dict"])
    # model.validation_step(None, None)

    data_irt = model.pack_irt_params()
    items_joint = list(zip(data, data_irt["items"]))
    items_joint.sort(
        key=lambda x: model.fn_utility(x[1], data_irt["systems"]),
        reverse=True
    )

    return [x[0] for x in items_joint]


METHODS = {
    "random": random,
    "avg": metric_avg,
    "var": metric_var,
    "metric_consistency": metric_consistency,
    "irt_diff": partial(irt, fn_utility="diff"),
    "irt_disc": partial(irt, fn_utility="disc"),
    "irt_feas": partial(irt, fn_utility="feas"),
    "irt_fic": partial(irt, fn_utility="fisher_information_content"),
}
