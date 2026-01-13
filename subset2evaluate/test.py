# %%

import numpy as np
import subset2evaluate.select_subset
import subset2evaluate.utils
import subset2evaluate.evaluate


# %%

def test_qe4pe_loader():
    data = subset2evaluate.utils.load_data_qe4pe(task="main")
    assert isinstance(data, dict)
    assert len(data) == 2
    assert len(data[("qe4pe", "eng-ita")]) == 324
    assert "i" in data[("qe4pe", "eng-ita")][0]
    assert "src" in data[("qe4pe", "eng-ita")][0]
    assert "tgt" in data[("qe4pe", "eng-ita")][0]
    assert "ref" in data[("qe4pe", "eng-ita")][0]
    assert "cost" in data[("qe4pe", "eng-ita")][0]
    assert "scores" in data[("qe4pe", "eng-ita")][0]
    assert len(data[("qe4pe", "eng-ita")][0]['scores']) == 12
    assert len(data[("qe4pe", "eng-nld")][0]['scores']) == 13

    data = subset2evaluate.utils.load_data_qe4pe(task="pretask")
    assert len(data) == 2
    assert len(data[("qe4pe", "eng-nld")]) == 38
    assert "scores" in data[("qe4pe", "eng-nld")][0]
    assert len(data[("qe4pe", "eng-ita")][0]['scores']) == 12
    assert len(data[("qe4pe", "eng-nld")][0]['scores']) == 13

    data = subset2evaluate.utils.load_data_qe4pe(task="posttask")
    assert len(data) == 2
    assert len(data[("qe4pe", "eng-ita")]) == 50
    assert "scores" in data[("qe4pe", "eng-ita")][0]
    assert len(data[("qe4pe", "eng-ita")][0]['scores']) == 11
    assert len(data[("qe4pe", "eng-nld")][0]['scores']) == 13


def test_biomqm_loader():
    data = subset2evaluate.utils.load_data_biomqm(split="test")
    assert isinstance(data, dict)
    assert len(data) == 15
    assert len(data[("biomqm", "en-de")]) == 757
    assert "i" in data[("biomqm", "en-de")][0]
    assert "src" in data[("biomqm", "en-de")][0]
    assert "tgt" in data[("biomqm", "en-de")][0]
    assert "ref" in data[("biomqm", "en-de")][0]
    assert "scores" in data[("biomqm", "en-de")][0]

    data = subset2evaluate.utils.load_data_biomqm(split="dev")
    assert len(data) == 10
    assert len(data[("biomqm", "en-de")]) == 255


def test_wmt_loader():
    data = subset2evaluate.utils.load_data("wmt/all", min_items=400, normalize=True)
    assert isinstance(data, dict)
    assert len(data) == 58
    assert len(data[("wmt23", "en-cs")]) == 1098
    assert len([k for k, v in data.items() if k[0].startswith("wmt23")]) == 10
    assert "src" in data[("wmt23", "en-cs")][0]
    assert "tgt" in data[("wmt23", "en-cs")][0]
    assert "scores" in data[("wmt23", "en-cs")][0]


def test_wmt_loader_mqm():
    data = subset2evaluate.utils.load_data_wmt(year="wmt24", langs="en-es", normalize=True)
    assert len(data) == 622
    data = subset2evaluate.utils.load_data_wmt(year="wmt24", langs="en-es", file_protocol="mqm", normalize=True)
    assert len(data) == 622


def test_wmt_method_random():
    data_new = subset2evaluate.select_subset.basic("wmt23/en-cs", method="random", seed=0)
    spa_new = subset2evaluate.evaluate.eval_spa(data_new, "wmt23/en-cs", metric="human")
    # random is usually random but we fix the seed
    assert abs(np.average(spa_new) - 0.854) < 0.20


def test_wmt_method_metric_var():
    data_new = subset2evaluate.select_subset.basic("wmt23/en-cs", method="metric_var", metric="MetricX-23-c")
    spa_new = subset2evaluate.evaluate.eval_spa(data_new, "wmt23/en-cs", metric="human")
    assert abs(np.average(spa_new) - 0.888) < 0.02


def test_wmt_method_diversity():
    data_new = subset2evaluate.select_subset.basic("wmt23/en-de", method="diversity", metric="BLEU")
    spa_new = subset2evaluate.evaluate.eval_spa(data_new, "wmt23/en-de", metric="human")
    assert abs(np.average(spa_new) - 0.927) < 0.02


def test_summeval_loader():
    data = subset2evaluate.utils.load_data("summeval")
    assert isinstance(data, list)
    assert len(data) == 100
    assert "tgt" in data[0]
    assert "scores" in data[0]


def test_summeval_method_random():
    data_new = subset2evaluate.select_subset.basic("summeval", method="random", seed=0)
    spa_new = subset2evaluate.evaluate.eval_spa(data_new, "summeval", metric="human_sum")
    # random is usually random but we fix the seed
    # it is a bit different on GitHub actions, therefore higher error margin
    assert abs(np.average(spa_new) - 0.920) < 0.20


def test_summeval_method_metric_var():
    data_new = subset2evaluate.select_subset.basic("summeval", method="metric_var", metric="coverage")
    spa_new = subset2evaluate.evaluate.eval_spa(data_new, "summeval", metric="human_sum")
    assert abs(np.average(spa_new) - 0.938) < 0.01


def test_summeval_method_diversity():
    data_new = subset2evaluate.select_subset.basic("summeval", method="diversity", metric="BLEU")
    spa_new = subset2evaluate.evaluate.eval_spa(data_new, "summeval", metric="human_sum")
    # it is a bit different on GitHub actions, therefore higher error margin
    assert abs(np.average(spa_new) - 0.927) < 0.20
