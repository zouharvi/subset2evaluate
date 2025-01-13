import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sentence_transformers import SentenceTransformer
import subset2evaluate.utils as utils
import logging
import argparse
import numpy as np


class MIRTDataset(Dataset):
    def __init__(self, data, metric='MetricX-23', embd_model_id='paraphrase-mpnet-base-v2'):
        self.item_ids = []
        self.system_ids = []
        self.scores = []
        self.embeddings = []
        self.item_map = {}
        self.system_map = {}

        self.encoder = SentenceTransformer(embd_model_id)
        all_embd = self.encoder.encode([example['src'] for example in data])

        for e_id, example in enumerate(data):
            if example['i'] not in self.item_map:
                self.item_map[example['i']] = example['i']
            for s_id, system in enumerate(example['scores']):
                if system not in self.system_map:
                    self.system_map[system] = len(self.system_map)
                self.item_ids.append(self.item_map[example['i']])
                self.system_ids.append(self.system_map[system])
                self.scores.append(example['scores'][system][metric])
                self.embeddings.append(all_embd[e_id])

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        return {
            'item_id': self.item_ids[idx],
            'system_id': self.system_ids[idx],
            'score': self.scores[idx],
            'embedding': self.embeddings[idx]
        }


class MultiDimensionalIRT(torch.nn.Module):
    def __init__(self, num_items, num_systems, dim):
        super(MultiDimensionalIRT, self).__init__()
        self.param_disc = torch.nn.Embedding(num_items, dim)
        self.param_ability = torch.nn.Embedding(num_systems, dim)
        self.param_diff = torch.nn.Embedding(num_items, 1)

    def forward(self, item_id, system_id):
        ability = self.param_ability(system_id)
        disc = self.param_disc(item_id)
        difficulty = self.param_diff(item_id)

        return torch.nn.functional.sigmoid(torch.sum(ability * disc, dim=-1, keepdim=True) - difficulty)

    def save_params(self):
        return {
            'disc': {k: v for k, v in enumerate(self.param_disc.weight.detach().numpy().tolist())},
            'diff': {k: v for k, v in enumerate(self.param_diff.weight.detach().numpy().tolist())},
            'ability': {k: v for k, v in enumerate(self.param_ability.weight.detach().numpy().tolist())}
        }


class MultiDimensionalIRTEmbedding(torch.nn.Module):
    def __init__(self, embd_size, num_systems, dim):
        super(MultiDimensionalIRTEmbedding, self).__init__()
        self.param_ability = torch.nn.Embedding(num_systems, dim)

        self.disc_layers = nn.Sequential(
            nn.Linear(embd_size, embd_size // 2),
            nn.ReLU(),
            nn.Linear(embd_size // 2, embd_size // 4),
            nn.ReLU(),
            nn.Linear(embd_size // 4, embd_size // 8),
            nn.ReLU(),
            nn.Linear(embd_size // 8, dim)
        )
        self.diff_layers = nn.Sequential(
            nn.Linear(embd_size, embd_size // 2),
            nn.ReLU(),
            nn.Linear(embd_size // 2, 1)
        )

    def forward(self, item_embd, system_id):
        disc = self.disc_layers(item_embd)
        diff = self.diff_layers(item_embd)
        ability = self.param_ability(system_id)

        return torch.nn.functional.sigmoid(torch.sum(ability * disc, dim=-1, keepdim=True) - diff)

    def save_params(self, all_embeddings):
        disc = self.disc_layers(all_embeddings).detach().numpy().tolist()
        diff = self.diff_layers(all_embeddings).detach().numpy().tolist()

        return {
            'disc': {k: v for k, v in enumerate(disc)},
            'diff': {k: v for k, v in enumerate(diff)},
            'ability': {k: v for k, v in enumerate(self.param_ability.weight.detach().numpy().tolist())}
        }


def train(batch_size, num_epoch, lr, embed, dim, split=0.9):
    all_data = utils.load_data_wmt(normalize=True)
    all_dataset = MIRTDataset(all_data)

    train_size = int(len(all_dataset) * split)
    val_size = len(all_dataset) - train_size
    dataset_train, dataset_val = random_split(all_dataset, [train_size, val_size])
    logging.info('{} train examples, {} test examples'.format(len(dataset_train), len(dataset_val)))

    if embed:
        irt_model = MultiDimensionalIRTEmbedding(embd_size=len(all_dataset.embeddings[0]), dim=dim, num_systems=len(all_dataset.system_map))
    else:
        irt_model = MultiDimensionalIRT(num_items=len(all_dataset.item_map), num_systems=len(all_dataset.system_map), dim=dim)

    train_dataloader = DataLoader(dataset_train, batch_size=batch_size)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(irt_model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        irt_model.train()
        train_loss = 0
        for batch_id, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()
            if embed:
                output = irt_model(item_embd=batch_data['embedding'], system_id=batch_data['system_id'])
            else:
                output = irt_model(item_id=batch_data['item_id'], system_id=batch_data['system_id'])
            loss = criterion(output, batch_data['score'].unsqueeze(-1))
            train_loss += loss
            loss.backward()
            optimizer.step()

        val_loss = 0
        if val_size > 0:
            irt_model.eval()
            for batch_data in val_dataloader:
                if embed:
                    output = irt_model(item_embd=batch_data['embedding'], system_id=batch_data['system_id'])
                else:
                    output = irt_model(item_id=batch_data['item_id'], system_id=batch_data['system_id'])
                loss = criterion(output, batch_data['score'].unsqueeze(-1))
                val_loss += loss

        logging.info('Epoch {}, train_loss {}, val loss {}'.format(epoch + 1, train_loss / train_size, val_loss / (val_size + 1e-5)))

    if embed:
        save_path = 'mirt_embd_param.jsonl'
    else:
        save_path = 'mirt_param.jsonl'

    if embed:
        params = irt_model.save_params(all_embeddings=torch.tensor(all_dataset.embeddings))
    else:
        params = irt_model.save_params()

    with open(save_path, 'w') as fp:
        fp.write(json.dumps(params))


def rank(param_file, ):

    def print_rank(sorted_data):
        all_acc = []
        for prop in utils.PROPS:
            subset = sorted_data[:int(len(all_data) * prop)]
            acc = utils.eval_subset_accuracy(subset, all_data)
            all_acc.append(acc)
            print('{}: {}'.format(prop, acc))
        print('avg acc: {}'.format(sum(all_acc) / len(all_acc)))

    with open(param_file, 'r') as fp:
        param = json.loads(fp.readlines()[0])

    for p in param:
        param[p] = {int(k): v for k, v in param[p].items()}

    all_data = utils.load_data_wmt(normalize=True)

    # random
    print('random')
    rnd_data = random.sample(all_data, len(all_data))
    print_rank(rnd_data)

    # difficulty
    print('diff')
    diff_sorted_data = sorted(all_data, key=lambda x: param['diff'][x['i']], reverse=True)
    print_rank(diff_sorted_data)

    # discrimination
    print('disc')
    disc_sorted_data = sorted(all_data, key=lambda x: sum(param['disc'][x['i']]), reverse=True)
    print_rank(disc_sorted_data)

    # fisher info
    print('fisher')
    fisher_info = {}  # item
    for item_id in param['disc']:
        fisher_info[item_id] = []
        for system_id in param['ability']:
            pred_score = np.sum(np.array(param['disc'][item_id]) * np.array(param['ability'][system_id])) - param['diff'][item_id]
            d_value = []  # each dim
            for d in range(32):
                fi = pred_score * (1 - pred_score) * param['disc'][item_id][d]**2
                d_value.append(fi)
            fisher_info[item_id].append(d_value)

    for item_id in fisher_info:
        fisher_info[item_id] = np.mean(fisher_info[item_id])

    fisher_sorted_data = sorted(all_data, key=lambda x: fisher_info[x['i']], reverse=True)
    print_rank(fisher_sorted_data)


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--embed', type=bool, default=True)
    parser.add_argument('--dim', type=int, default=32)

    args = parser.parse_args()

    # train(batch_size=args.batch_size, lr=args.lr, num_epoch=args.epoch, embed=args.embed, dim=args.dim, split=1)

    rank('mirt_param.jsonl')
    # rank('mirt_embd_param.jsonl')
