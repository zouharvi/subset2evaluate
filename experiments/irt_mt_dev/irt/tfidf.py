import torch
import torch.utils
from irt_mt_dev.irt.base import IRTModelBase
from sklearn.feature_extraction.text import TfidfVectorizer

class IRTModelTFIDF(IRTModelBase):
    def __init__(self, items, systems, **kwargs):
        super().__init__(systems=systems, **kwargs)

        encoder = TfidfVectorizer(max_features=768)
        self.text_src = torch.nn.Parameter(
            torch.tensor(encoder.fit_transform([item["src"] for item in items]).toarray()).float(),
            requires_grad=False
        )

        # normally distribute at the beginning
        self.param_disc = torch.nn.Linear(768, 1)
        self.param_diff = torch.nn.Linear(768, 1)
        self.param_feas = torch.nn.Linear(768, 1)

        self.len_items = len(items)
    
    def get_irt_params(self, i_item, name):
        if name == "disc":
            return self.param_disc(self.text_src[i_item,:]).flatten()
        elif name == "diff":
            return self.param_diff(self.text_src[i_item,:]).flatten()
        elif name == "feas":
            return self.param_feas(self.text_src[i_item,:]).flatten()
        
    def pack_irt_params_items(self):
        return [
            {
                "disc": self.get_irt_params(i_item, name="disc").item(),
                "diff": self.get_irt_params(i_item, name="diff").item(),
                "feas": self.get_irt_params(i_item, name="feas").item(),
            }
            for i_item in range(self.len_items)
        ]