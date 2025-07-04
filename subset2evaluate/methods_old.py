# These are some subset selection methods that are not polished enough to be used in practice
from typing import Any, Callable, List, Dict
import subset2evaluate.utils as utils
import subset2evaluate.evaluate


def nn_irt(data, metric, **kwargs):
    raise NotImplementedError("This method is not yet implemented for use.")
    import torch
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

    models = list(data[0]["scores"].keys())

    def wrangle_data(data_local):
        data_out = []
        for line in data_local:
            for model, model_v in line["scores"].items():
                data_out.append({
                    "agent_name": model,
                    "agent_id": models.index(model),
                    "agent_type": "general",
                    "query_id": line["i"],
                    "query_rep": torch.nn.functional.one_hot(torch.tensor([line["i"]]), num_classes=len(data)).float(),
                    "ruling": model_v[metric],
                })
        return data_out

    def create_agent_indexer_from_dataset(
        dataset_or_path: str | Sequence[Dict[str, Any]],
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
    agent_indexer = create_agent_indexer_from_dataset(data_train + data_test)
    train_collator = collators.CaimiraCollator(agent_indexer, is_training=True)

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
        # NOTE: we define max_epochs twice?
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

        c_reg_skill: float = 1e-6
        c_reg_difficulty: float = 1e-6
        c_reg_relevance: float = 1e-6

    class CaimiraConfig(IrtModelConfig):
        # Number of dimensions in item embeddings
        # NOTE: turn this into real embeddings
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

        n_agents: int = len(models)
        # 1 for now because we don't know what it really is
        n_agent_types: int = 1
        n_dim: int = 32
        fit_guess_bias: float = False
        # NOTE: turn off?
        fit_agent_type_embeddings: bool = True

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


def our_irt(data, metric, **kwargs):
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

    models = list(data[0]["scores"].keys())

    ModelClass = {
        "scalar": IRTModelScalar,
        "tfidf": IRTModelTFIDF,
        "embd": IRTModelEmbd,
    }[kwargs["model"]]
    model = ModelClass(data, models, data_old=data, **kwargs)

    data_flat = [
        ((sent_i, model_i), sent["scores"][model][metric])
        for sent_i, sent in enumerate(data)
        for model_i, model in enumerate(models)
    ]

    # NOTE: in the future run first training with dev set to find out the best epoch count
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

    return [
        model.fn_utility(x, data_irt["models"])
        for x in data_irt["items"]
    ]


def premlp_other(data, data_train, fn_utility: Callable, **kwargs) -> List[float]:
    # turn off warnings from sentence-transformers
    import warnings
    warnings.filterwarnings('ignore')
    import sentence_transformers
    import sklearn.neural_network

    embd_model = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L12-v2")
    data_x_train = embd_model.encode([line["src"] for line in data_train])
    data_y_train = [fn_utility(line) for line in data_train]

    model = sklearn.neural_network.MLPRegressor(
        hidden_layer_sizes=(128, 16),
        max_iter=1000,
        verbose=False,
    )
    model.fit(data_x_train, data_y_train)
    data_x_test = embd_model.encode([line["src"] for line in data])
    data_y_test = model.predict(data_x_test)

    warnings.resetwarnings()
    return list(data_y_test)