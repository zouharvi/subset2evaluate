import torch
import numpy as np
import irt_mt_dev.utils as utils
from functools import partial


def random(data, **kwargs):
    import random
    # random.Random(0).shuffle(data)
    random.shuffle(data)
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


def _fn_information_content(item_irt, data_irt):
    information = 0
    for theta in data_irt["systems"].values():
        prob = utils.pred_irt(
            theta,
            item_irt
        )
        information += prob*(1-prob)*(item_irt["disc"]**2)
    return information

def _fn_experimental(item_old, item_irt, data_irt):
    information = 0
    for system, theta in data_irt["systems"].items():
        prob = utils.pred_irt(
            theta,
            item_irt
        )
        error = (item_old["scores"][system]["MetricX-23-c"] - prob)**2
        information += prob*(1-prob)*(item_irt["disc"]**2)/error
    return information


def _our_irt(data, **kwargs):
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
    import wandb
    import os
    os.environ["WANDB_SILENT"] = "true"
    from lightning.pytorch.loggers import WandbLogger


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
            print(exception)
            exit()

        
    trainer = L.Trainer(
        max_epochs=2000,
        check_val_every_n_epoch=50,
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        # logger=False,
        callbacks=[
            EarlyStopping(monitor="cluster_count_metric", patience=20, verbose=True, mode="max"),
            ModelCheckpoint(filename='best_model', monitor='cluster_count_metric', mode='max', save_top_k=1),
            PropagateExitCallback(),
        ],
        logger=WandbLogger(project="irt-mt-dev"),
    )

    trainer.fit(
        model=model,
        train_dataloaders=data_train,
        val_dataloaders=data_val,
    )
    wandb.finish()

    # reload best model
    model.eval()
    # model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path, weights_only=True)["state_dict"])
    # model.validation_step(None, None)
    # data_irt = model.pack_irt_params()

    best_val_step = max(range(len(model.results_log)), key=lambda i: model.results_log[i]["subset_consistency_accuracy_metric"])
    # print("Best validation step was:", best_val_step)
    data_irt = model.params_log[best_val_step]

    items_joint = list(zip(data, data_irt["items"]))
    items_joint.sort(
        key=lambda x: model.fn_utility(x[1], data_irt["systems"]),
        reverse=True
    )

    return [x[0] for x in items_joint]

def pyirt(data, return_model=False, load_model=None, model="4pl_score", dropout=0.25, **kwargs):
    import py_irt
    import py_irt.config
    import py_irt.dataset
    import py_irt.io
    import py_irt.training
    import pandas as pd


    systems = list(data[0]["scores"].keys())

    if load_model is not None:
        data_irt = load_model
    else:
        # we need median binarization if we are not using 4pl_score model
        median = np.median([
            system_v[kwargs["metric"]]
            for line in data
            for system_v in line["scores"].values()
        ])
        dataset = pd.DataFrame({
            "system": systems,
            **{
                f"item_{line['i']}": [
                    line["scores"][system][kwargs["metric"]]
                    if "_score" in model else
                    line["scores"][system][kwargs["metric"]] >= median
                    for system in systems
                ]
                for line in data
            }
        })

        embeddings = None
        if "amortized_" in model:
            import sentence_transformers
            embd_model = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L12-v2")
            embeddings = embd_model.encode([line["src"] for line in data])
            embeddings = {f"item_{line['i']}": emb.tolist() for line, emb in zip(data, embeddings)}
            del embd_model

        dataset = py_irt.dataset.Dataset.from_pandas(
            dataset,
            subject_column="system",
            item_columns=[f"item_{line['i']}" for line in data],
            embeddings=embeddings,
        )

        config = py_irt.config.IrtConfig(
            model_type=model,
            log_every=10,
            dropout=dropout,
            priors="hiearchical",
            seed=0,
            deterministic=True,
        )
        trainer = py_irt.training.IrtModelTrainer(
            config=config,
            data_path=None,
            dataset=dataset,
            verbose=True
        )
        trainer.train(epochs=kwargs["epochs"], device='cuda')

        params = trainer.best_params

        # this flipping will not affect the predictions
        # if np.average(params["disc"]) < 0 and np.average(params["ability"]) < 0:
        #     params["disc"] = -np.array(params["disc"])
        #     params["ability"] = -np.array(params["ability"])
        #     params["diff"] = -np.array(params["diff"])
        
        # normalize naming
        if "lambdas" in params:
            params["feas"] = params.pop("lambdas")
    

        # TODO: cross-check make sure that we do the predictions as the models were trained
        # 3PL/4PL
        if "feas" in params:
            data_irt = {
                "systems": {sys: sys_v for sys, sys_v in zip(systems, params["ability"])},
                "items": [
                    {"disc": disc, "diff": diff, "feas": feas}
                    for disc, diff, feas in zip(
                        params["disc"],
                        params["diff"],
                        params["feas"],
                    )
                ]
            }
        elif "disc" in params:
            data_irt = {
                "systems": {sys: sys_v for sys, sys_v in zip(systems, params["ability"])},
                "items": [
                    {"disc": disc, "diff": diff}
                    for disc, diff in zip(
                        params["disc"],
                        params["diff"],
                    )
                ]
            }
        else:
            data_irt = {
                "systems": {sys: sys_v for sys, sys_v in zip(systems, params["ability"])},
                "items": [
                    {"diff": diff}
                    for diff in zip(
                        params["diff"],
                    )
                ]
            }

    def fn_irt_utility(item_old, item_irt, data_irt, fn_utility):
        if fn_utility == "experimental":
            return _fn_experimental(item_old, item_irt, data_irt)
        elif fn_utility == "fisher_information_content":
            return _fn_information_content(item_irt, data_irt)
        elif fn_utility == "diff":
            return -item_irt["diff"]
        elif fn_utility == "disc":
            return -item_irt["disc"]
        elif fn_utility == "diffdisc":
            return item_irt["diff"]*item_irt["disc"]
        elif fn_utility == "feas":
            return item_irt["feas"]

    items_joint = list(zip(data, data_irt["items"]))
    items_joint.sort(
        key=lambda x: fn_irt_utility(x[0], x[1], data_irt, kwargs["fn_utility"]),
        reverse=True
    )

    items = [x[0] for x in items_joint]

    if return_model:
        return items, data_irt
    else:
        return items

def _nn_irt(data, **kwargs):
    import neural_irt.train
    from neural_irt.lit_module import IrtLitModule
    from neural_irt.data import collators, datasets
    from neural_irt.configs.common import IrtModelConfig
    from torch.utils import data as torch_data
    import wandb
    from lightning.pytorch.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from sklearn.model_selection import train_test_split
    from typing import Any, Optional, Sequence
    from pydantic import BaseModel

    BATCH_SIZE = 256

    systems = list(data[0]["scores"].keys())

    def wrangle_data(data_local):
        data_out = []
        for line in data_local:
            for sys, sys_v in line["scores"].items():
                data_out.append({
                    "agent_name": sys,
                    "agent_id": systems.index(sys),
                    "agent_type": "general",
                    "query_id": line["i"],
                    "query_rep": torch.nn.functional.one_hot(torch.tensor([line["i"]]), num_classes=len(data)).float(),
                    "ruling": sys_v[kwargs["metric"]],
                })
        return data_out
    

    def create_agent_indexer_from_dataset(
        dataset_or_path: str | Sequence[dict[str, Any]],
    ) -> neural_irt.train.AgentIndexer:
        dataset = dataset_or_path
        if isinstance(dataset, str):
            dataset = datasets.load_as_hf_dataset(dataset_or_path)

        # NOTE: this was entry["id"] and entry["type"] before
        agent_ids = [entry["agent_id"] for entry in dataset]
        agent_types = list({entry["agent_type"] for entry in dataset})
        agent_type_map = {entry["agent_id"]: entry["agent_type"] for entry in dataset}
        return neural_irt.train.AgentIndexer(agent_ids, agent_types, agent_type_map)

    
    data_train, data_test = train_test_split(data, test_size=0.1, random_state=0)
    data_train = wrangle_data(data_train)
    data_test = wrangle_data(data_test)
    agent_indexer = create_agent_indexer_from_dataset(data_train+data_test)
    train_collator = collators.CaimiraCollator(agent_indexer, is_training=True)
    val_collator = collators.CaimiraCollator(agent_indexer, is_training=False)

    # TODO: very likely will fail here because the data is not wrangled properly
    train_loader = torch_data.DataLoader(
        data_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=1,
    )
    val_loaders_dict = {
        "val": torch_data.DataLoader(
        data_test,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=1,
    )
    }
    val_loader_names = list(val_loaders_dict.keys())
    val_loaders = [val_loaders_dict[name] for name in val_loader_names]


    class TrainerConfig(BaseModel):
        # Train time
        # TODO: we define max_epochs twice?
        max_epochs: int = 100
        max_steps: Optional[int] = None
        sampler: Optional[str] = None
        batch_size: int = BATCH_SIZE

        # Optimizer
        optimizer: str = "Adam"  # [Adam, RMSprop, SGD]
        learning_rate: float = 1e-3
        cyclic_lr: bool = False

        second_optimizer: str = "SGD"
        second_learning_rate: float = 5e-4
        second_optimizer_start_epoch: Optional[int] = 75

        freeze_bias_after: Optional[int] = None

        ckpt_savedir: str = "./checkpoints/irt"

        c_reg_skill : float = 1e-6
        c_reg_difficulty : float = 1e-6
        c_reg_relevance : float = 1e-6

    class CaimiraConfig(IrtModelConfig):
        # Number of dimensions in item embeddings
        # TODO: turn this into real embeddings
        n_dim_item_embed: int = len(data)

        # Number of dimensions for the agent embedding
        rel_mode: str = "linear"  # [linear, mlp]
        dif_mode: str = "linear"  # [linear, mlp]

        # Number of hidden units for the MLPs if mode is mlp
        n_hidden_dif: int = 128
        n_hidden_rel: int = 128

        # Sparsity controls for importance [only used if fit_importance is True]
        # Temperature for importance
        rel_temperature: float = 0.5
        fast: bool = False

        @property
        def arch(self):
            return "caimira"
        
        n_agents: int = len(systems)
        # 1 for now because we don't know what it really is
        n_agent_types : int = 1
        n_dim : int = 32
        fit_guess_bias : float = False
        # TODO: turn off?
        fit_agent_type_embeddings : bool = True


    model = IrtLitModule(
        trainer_config=TrainerConfig(),
        model_or_config=CaimiraConfig(),
        val_dataloader_names=val_loader_names,
        agent_indexer=agent_indexer,
    )

    train_logger = None
    train_logger = WandbLogger(
        project="irt-mt-dev",
        name="nnirt base",
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val/acc",
        mode="max",  # Error: This should be "max" instead of "min" for accuracy
        auto_insert_metric_name=False,
        filename="epoch={epoch}-acc={val/acc:.2f}",
    )
    checkpoint_callback.FILE_EXTENSION = ""
    trainer = neural_irt.train.CaimiraTrainer(
        max_epochs=kwargs["max_epochs"],
        accelerator="auto",
        logger=train_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loaders)
    loaded_model = IrtLitModule.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    print(loaded_model)
    wandb.save()

def cometsrc(data, model_path, reverse=False, **kwargs):
    import comet

    model = comet.load_from_checkpoint(model_path)
    scores = model.predict([
        {"src": line["src"]}
        for line in data
    ]).scores

    data_w_score = list(zip(data, scores))
    data_w_score.sort(key=lambda x: x[1], reverse=reverse)

    return [x[0] for x in data_w_score]

METHODS = {
    "random": random,
    "avg": metric_avg,
    "var": metric_var,
    "metric_consistency": metric_consistency,
    "pyirt_diff": partial(pyirt, fn_utility="diff"),
    "pyirt_disc": partial(pyirt, fn_utility="disc"),
    "pyirt_diffdisc": partial(pyirt, fn_utility="diffdisc"),
    "pyirt_feas": partial(pyirt, fn_utility="feas"),
    "pyirt_fic": partial(pyirt, fn_utility="fisher_information_content"),
    "pyirt_exp": partial(pyirt, fn_utility="experimental"),
    "_our_irt_diff": partial(_our_irt, fn_utility="diff"),
    "_our_irt_disc": partial(_our_irt, fn_utility="disc"),
    "_our_irt_feas": partial(_our_irt, fn_utility="feas"),
    "_our_irt_fic": partial(_our_irt, fn_utility="fisher_information_content"),
    "_nn_irt_fic": partial(_nn_irt, fn_utility="fisher_information_content"),
    # second epoch is always the best
    "precomet_var": partial(cometsrc, model_path="/home/vilda/comet-src/lightning_logs/version_0/checkpoints/epoch=1-step=3124-val_pearson=0.024.ckpt", reverse=False),
    "precomet_avg": partial(cometsrc, model_path="/home/vilda/comet-src/lightning_logs/version_1/checkpoints/epoch=1-step=3124-val_pearson=0.047.ckpt", reverse=False),
}