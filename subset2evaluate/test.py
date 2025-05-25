# %%

import numpy as np
import subset2evaluate


def test_wmt_loader():
    data = subset2evaluate.utils.load_data("wmt/all", min_items=400)
    assert isinstance(data, dict)
    assert len(data) == 58
    assert len(data[("wmt23", "en-cs")]) == 1098
    assert len([k for k, v in data.items() if k[0].startswith("wmt23")]) == 10
    assert "src" in data[("wmt23", "en-cs")][0]
    assert "tgt" in data[("wmt23", "en-cs")][0]
    assert "scores" in data[("wmt23", "en-cs")][0]


def test_wmt_loader_mqm():
    data = subset2evaluate.utils.load_data_wmt(year="wmt24", langs="en-es")
    assert len(data) == 634
    data = subset2evaluate.utils.load_data_wmt(year="wmt24", langs="en-es", file_protocol="mqm")
    assert len(data) == 622


def test_wmt_method_random():
    data_new = subset2evaluate.select_subset.basic("wmt23/en-cs", method="random", seed=0)
    clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, "wmt23/en-cs", metric="human")
    # random is usually random but we fix the seed
    assert abs(np.average(clu_new) - 1.4000) < 0.01
    assert abs(np.average(cor_new) - 0.7814) < 0.01


def test_wmt_method_metric_var():
    data_new = subset2evaluate.select_subset.basic("wmt23/en-cs", method="metric_var", metric="MetricX-23-c")
    clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, "wmt23/en-cs", metric="human")
    assert abs(np.average(clu_new) - 1.8000) < 0.01
    assert abs(np.average(cor_new) - 0.8450) < 0.01


def test_wmt_method_diversity():
    data_new = subset2evaluate.select_subset.basic("wmt23/en-de", method="diversity", metric="BLEU")
    clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, "wmt23/en-de", metric="human")
    assert abs(np.average(clu_new) - 2.3000) < 0.01
    assert abs(np.average(cor_new) - 0.9328) < 0.01


def test_summeval_loader():
    data = subset2evaluate.utils.load_data("summeval")
    assert isinstance(data, list)
    assert len(data) == 100
    assert "tgt" in data[0]
    assert "scores" in data[0]


def test_summeval_method_random():
    data_new = subset2evaluate.select_subset.basic("summeval", method="random", seed=0)
    clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, "summeval", metric="human_sum")
    # random is usually random but we fix the seed
    # it is a bit different on GitHub actions, therefore higher error margin
    assert abs(np.average(clu_new) - 1.6000) < 0.2
    assert abs(np.average(cor_new) - 0.9294) < 0.2


def test_summeval_method_metric_var():
    data_new = subset2evaluate.select_subset.basic("summeval", method="metric_var", metric="coverage")
    clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, "summeval", metric="human_sum")
    assert abs(np.average(clu_new) - 2.5000) < 0.01
    assert abs(np.average(cor_new) - 0.9682) < 0.01


def test_summeval_method_diversity():
    data_new = subset2evaluate.select_subset.basic("summeval", method="diversity", metric="BLEU")
    clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, "summeval", metric="human_sum")
    # it is a bit different on GitHub actions, therefore higher error margin
    assert abs(np.average(clu_new) - 2.6000) < 0.2
    assert abs(np.average(cor_new) - 0.9364) < 0.2



# %%
import tqdm
import subset2evaluate.utils as utils
import numpy as np
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import importlib
importlib.reload(subset2evaluate.evaluate)

data_old_all = list(utils.load_data_wmt_test().items())
PROPS = np.linspace(0.05, 0.5, 10)

# METRICS_ALL = ['human', 'metametrics_mt_mqm_hybrid_kendall', 'metametrics_mt_mqm_same_source_targ', 'BERTScore', 'sentinel-src-mqm', 'chrF', 'metametrics_mt_mqm_qe_same_source_t', 'XLsimMqm', 'MetricX-24-Hybrid', 'XLsimDA', 'YiSi-1', 'spBLEU', 'PrismRefMedium', 'monmonli', 'COMET-22', 'MetricX-24-Hybrid-QE', 'BLEU', 'sentinel-ref-mqm', 'XCOMET', 'PrismRefSmall', 'metametrics_mt_mqm_kendall', 'CometKiwi', 'MetricX-24', 'gemba_esa', 'XCOMET-QE', 'metametrics_mt_mqm_qe_kendall.seg.s', 'sentinel-cand-mqm', 'damonmonli', 'MetricX-24-QE', 'CometKiwi-XXL', 'chrfS', 'BLEURT-20']
METRICS_ALL = ['human', 'MetricX-23-QE', 'mre-score-labse-regular', 'MetricX-23', 'chrF', 'COMET', 'Random-sysname', 'f200spBLEU', 'tokengram_F', 'GEMBA-MQM', 'YiSi-1', 'embed_llama', 'XCOMET-XXL', 'BLEU', 'prismRef', 'eBLEU', 'cometoid22-wmt22', 'KG-BERTScore', 'MetricX-23-QE-c', 'XCOMET-XL', 'CometKiwi', 'XCOMET-QE-Ensemble', 'MetricX-23-QE-b', 'CometKiwi-XL', 'MS-COMET-QE-22', 'MetricX-23-c', 'prismSrc', 'cometoid22-wmt21', 'XCOMET-Ensemble', 'BERTscore', 'XLsim', 'CometKiwi-XXL', 'cometoid22-wmt23', 'BLEURT-20', 'MetricX-23-b']

for metric in tqdm.tqdm(METRICS_ALL):
    points_y_spa = []
    points_y_top = []
    for data_old_name, data_old in data_old_all:
        spa_new, top_new = subset2evaluate.evaluate.eval_spa(
            subset2evaluate.select_subset.basic(data_old, method="metric_cons", metric=metric),
            data_old,
            metric="human",
            props=PROPS,
        )
        points_y_spa.append(spa_new)
        points_y_top.append(top_new)
    print(metric, f"{np.average(points_y_spa):.1%}", f"{np.average(points_y_top):.1%}")

# metric_cons: 
    
# %%

points_y_spa = []
points_y_top = []
for _ in tqdm.tqdm(range(10)):
    for data_old_name, data_old in data_old_all:
        spa_new, top_new = subset2evaluate.evaluate.eval_spa(
            subset2evaluate.select_subset.basic(data_old, method="random"),
            data_old,
            metric="human",
            props=PROPS,
        )
        points_y_spa.append(spa_new)
        points_y_top.append(top_new)
print("random", f"{np.average(points_y_spa):.1%}", f"{np.average(points_y_top):.1%}")

# random 90.2%