# %%

import numpy as np
import tqdm
import subset2evaluate.utils as utils
import subset2evaluate.utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import multiprocessing
import collections

data_old_all = list({
    "en-cs": utils.load_data_wmt(year="wmt24", langs="en-cs", normalize=True),
    "cs-uk": utils.load_data_wmt(year="wmt24", langs="cs-uk", normalize=True),
    "en-de": utils.load_data_wmt(year="wmt24", langs="en-de", normalize=True),
    "en-es": utils.load_data_wmt(year="wmt24", langs="en-es", normalize=True),
    "en-hi": utils.load_data_wmt(year="wmt24", langs="en-hi", normalize=True),
    "en-ja": utils.load_data_wmt(year="wmt24", langs="en-ja", normalize=True),
    "en-ru": utils.load_data_wmt(year="wmt24", langs="en-ru", normalize=True),
    "en-uk": utils.load_data_wmt(year="wmt24", langs="en-uk", normalize=True),
    "en-zh": utils.load_data_wmt(year="wmt24", langs="en-zh", normalize=True),
    "ja-zh": utils.load_data_wmt(year="wmt24", langs="ja-zh", normalize=True),
}.items())

print({k: len(v) for k, v in data_old_all})

for _, data_old in data_old_all:
    print(_, collections.Counter([x["domain"] for x in data_old]))

# %%
with multiprocessing.Pool(len(data_old_all)) as pool:
    clucor_precomputed_values = pool.starmap(
        subset2evaluate.evaluate.precompute_randnorm,
        [(x[1], 10, "human", 2) for x in data_old_all]
    )
clucor_precomputed = dict(zip([x[0] for x in data_old_all], clucor_precomputed_values))

# %%
import itertools

# TODO: balance domains here maybe?
for method_kwargs in [
    dict(method="random"),
    dict(method="random"),
    dict(method="random"),
    dict(method="random"),
    dict(method="metric_var", metric="MetricX-24"),
    dict(method="metric_avg", metric="MetricX-24"),
    dict(method="diversity_unigram"),
    dict(method="diversity_bleu"),
    dict(method="diversity_chrf"),
    dict(method="pyirt_diffdisc", metric="MetricX-24"),
    dict(method="precomet_diversity"),
    dict(method="precomet_diffdisc"),
]:
    par_clu_all = []
    par_acc_all = []
    for data_name, data_old in tqdm.tqdm(data_old_all):
        data_new = subset2evaluate.select_subset.basic(data_old, **method_kwargs)

        # # balance domains
        # domains = collections.defaultdict(list)
        # for line in data_new:
        #     domains[line["domain"]].append(line)

        # data_aggregated = [
        #     sorted(v, key=lambda x: x["subset2evaluate_utility"], reverse=True)
        #     for k, v in domains.items()
        # ]
        # data_new = [doc for docs in itertools.zip_longest(*data_aggregated) for doc in docs]
        # data_new = [line for line in data_new if line is not None]

        # respect document units
        # documents = collections.defaultdict(list)
        # for line in data_new:
        #     documents[line["doc"]].append(line)
        # data_new = sorted(
        #     documents.values(),
        #     key=lambda doc_v: np.average([line["subset2evaluate_utility"] for line in doc_v]),
        #     reverse=True,
        # )
        # data_new = [line for doc_v in data_new for line in doc_v]

        # balance domains AND respect document units
        documents = collections.defaultdict(list)
        for line in data_new:
            documents[line["doc"]].append(line)

        domains = collections.defaultdict(list)
        for doc in documents.values():
            domains[doc[0]["domain"]].append(doc)

        data_aggregated = [
            sorted(v, key=lambda x: np.average([l["subset2evaluate_utility"] for l in x]), reverse=True)
            for k, v in domains.items()
        ]
        data_new = [doc for docs in itertools.zip_longest(*data_aggregated) for doc in docs]
        data_new = [line for line in data_new if line is not None]
        data_new = [line for doc_v in data_new for line in doc_v]
        

        # par_clu, par_acc = subset2evaluate.evaluate.eval_clucor_randnorm(
        #     subset2evaluate.select_subset.basic(data_old, **method_kwargs),
        #     data_old,
        #     clucor_precomputed=clucor_precomputed[data_name],
        # )
        par_clu, par_acc = subset2evaluate.evaluate.eval_clucor(
            data_new,
            data_old,
        )
        par_clu_all.append(np.average(par_clu))
        par_acc_all.append(np.average(par_acc))
    # print(f'{method_kwargs["method"]:<15}', f"CLU: {np.average(par_clu_all):.1%} | ACC: {np.average(par_acc_all):.1%}")
    print(f'{method_kwargs["method"]:<15}', f"CLU: {np.average(par_clu_all):.2f} | ACC: {np.average(par_acc_all):.1%}")