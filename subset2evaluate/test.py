import numpy as np
import subset2evaluate


def test_wmt_loader():
    data = subset2evaluate.utils.load_data("wmt23/all")
    assert isinstance(data, dict)
    assert len(data) == 33
    assert len(data[("wmt23", "en-cs")]) == 1098
    assert "src" in data[("wmt23", "en-cs")][0]
    assert "tgt" in data[("wmt23", "en-cs")][0]
    assert "scores" in data[("wmt23", "en-cs")][0]


def test_wmt_method_random():
    data_new = subset2evaluate.select_subset.run_select_subset("wmt23/en-cs", method="random", seed=0)
    clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_cluacc(data_new, "wmt23/en-cs", metric="human")
    # random is usually random but we fix the seed
    assert abs(np.average(clu_new) - 1.4000) < 0.01
    assert abs(np.average(acc_new) - 0.8104) < 0.01


def test_wmt_method_metric_var():
    data_new = subset2evaluate.select_subset.run_select_subset("wmt23/en-cs", method="metric_var", metric="MetricX-23-c")
    clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_cluacc(data_new, "wmt23/en-cs", metric="human")
    assert abs(np.average(clu_new) - 1.8000) < 0.01
    assert abs(np.average(acc_new) - 0.8552) < 0.01


def test_wmt_method_diversity():
    data_new = subset2evaluate.select_subset.run_select_subset("wmt23/en-de", method="diversity_bleu")
    clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_cluacc(data_new, "wmt23/en-de", metric="human")
    assert abs(np.average(clu_new) - 2.3000) < 0.01
    assert abs(np.average(acc_new) - 0.9152) < 0.01


def test_summeval_loader():
    data = subset2evaluate.utils.load_data("summeval")
    assert isinstance(data, list)
    assert len(data) == 100
    assert "tgt" in data[0]
    assert "scores" in data[0]


def test_summeval_method_random():
    data_new = subset2evaluate.select_subset.run_select_subset("summeval", method="random", seed=0)
    clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_cluacc(data_new, "summeval", metric="human_all")
    # random is usually random but we fix the seed
    # it is a bit different on GitHub actions, therefore higher error margin
    assert abs(np.average(clu_new) - 1.6000) < 0.2
    assert abs(np.average(acc_new) - 0.9279) < 0.2


def test_summeval_method_metric_var():
    data_new = subset2evaluate.select_subset.run_select_subset("summeval", method="metric_var", metric="coverage")
    clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_cluacc(data_new, "summeval", metric="human_all")
    assert abs(np.average(clu_new) - 2.3000) < 0.01
    assert abs(np.average(acc_new) - 0.9220) < 0.01


def test_summeval_method_diversity():
    data_new = subset2evaluate.select_subset.run_select_subset("summeval", method="diversity_bleu")
    clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_cluacc(data_new, "summeval", metric="human_all")
    # it is a bit different on GitHub actions, therefore higher error margin
    assert abs(np.average(clu_new) - 2.9000) < 0.2
    assert abs(np.average(acc_new) - 0.8934) < 0.2
