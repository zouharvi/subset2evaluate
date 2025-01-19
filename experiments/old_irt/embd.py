# flake8: noqa E402
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.utils
from base import IRTModelBase
from sentence_transformers import SentenceTransformer


class IRTModelEmbd(IRTModelBase):
    def __init__(self, data, models, **kwargs):
        super().__init__(models=models, **kwargs)

        encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        self.text_src = torch.nn.Parameter(
            torch.tensor(encoder.encode([item["src"] for item in data])),
            requires_grad=False,
        )

        # normally distribute at the beginning
        fn = lambda: torch.nn.Sequential(
            # torch.nn.Linear(384, 384),
            # torch.nn.ReLU(),
            # torch.nn.Linear(384, 384),
            # torch.nn.ReLU(),
            torch.nn.Linear(384, 1),
        )
        self.param_disc = fn()
        self.param_diff = fn()
        self.param_feas = fn()

        self.len_items = len(data)

    def get_irt_params(self, i_item, name):
        if name == "disc":
            return self.param_disc(self.text_src[i_item, :]).flatten()
        elif name == "diff":
            return self.param_diff(self.text_src[i_item, :]).flatten()
        elif name == "feas":
            return self.param_feas(self.text_src[i_item, :]).flatten()

    def pack_irt_params_items(self):
        return [
            {
                "disc": self.get_irt_params(i_item, name="disc").item(),
                "diff": self.get_irt_params(i_item, name="diff").item(),
                "feas": self.get_irt_params(i_item, name="feas").item(),
            }
            for i_item in range(self.len_items)
        ]
