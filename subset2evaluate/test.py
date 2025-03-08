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