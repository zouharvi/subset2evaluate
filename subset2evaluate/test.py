import numpy as np
import sys
sys.path.append('/Users/cuipeng/Documents/Projects/subset2evaluate')
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
    data_new = subset2evaluate.select_subset.basic("wmt23/en-cs", method="random", seed=0)
    clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, "wmt23/en-cs", metric="human")
    # random is usually random but we fix the seed
    assert abs(np.average(clu_new) - 1.4000) < 0.01
    assert abs(np.average(acc_new) - 0.8104) < 0.01


def test_wmt_method_metric_var():
    data_new = subset2evaluate.select_subset.basic("wmt23/en-cs", method="metric_var", metric="MetricX-23-c")
    clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, "wmt23/en-cs", metric="human")
    assert abs(np.average(clu_new) - 1.8000) < 0.01
    assert abs(np.average(acc_new) - 0.8552) < 0.01


def test_wmt_method_diversity():
    data_new = subset2evaluate.select_subset.basic("wmt23/en-de", method="diversity_bleu")
    clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, "wmt23/en-de", metric="human")
    assert abs(np.average(clu_new) - 2.3000) < 0.01
    assert abs(np.average(acc_new) - 0.9152) < 0.01


def test_summeval_loader():
    data = subset2evaluate.utils.load_data("summeval")
    assert isinstance(data, list)
    assert len(data) == 100
    assert "tgt" in data[0]
    assert "scores" in data[0]


def test_summeval_method_random():
    data_new = subset2evaluate.select_subset.basic("summeval", method="random", seed=0)
    clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, "summeval", metric="human_mul")
    # random is usually random but we fix the seed
    # it is a bit different on GitHub actions, therefore higher error margin
    assert abs(np.average(clu_new) - 1.6000) < 0.2
    assert abs(np.average(acc_new) - 0.9279) < 0.2


def test_summeval_method_metric_var():
    data_new = subset2evaluate.select_subset.basic("summeval", method="metric_var", metric="coverage")
    clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, "summeval", metric="human_mul")
    assert abs(np.average(clu_new) - 2.3000) < 0.01
    assert abs(np.average(acc_new) - 0.9220) < 0.01


def test_summeval_method_diversity():
    data_new = subset2evaluate.select_subset.basic("summeval", method="diversity_bleu")
    clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, "summeval", metric="human_mul")
    # it is a bit different on GitHub actions, therefore higher error margin
    assert abs(np.average(clu_new) - 2.9000) < 0.2
    assert abs(np.average(acc_new) - 0.8934) < 0.2


def test_custom(data_file, method, method_metric, human_metric):
    # e2e challenge
    data_full = subset2evaluate.utils.load_data(data_file)
    data_rnd = subset2evaluate.select_subset.basic(data_full, method='random')
    clu_rnd, acc_rnd = subset2evaluate.evaluate.eval_cluacc(data_rnd, data_full, metric=human_metric)

    data_sorted = subset2evaluate.select_subset.basic(data_full, method=method, metric=method_metric)
    clu_method, acc_method = subset2evaluate.evaluate.eval_cluacc(data_sorted, data_full, metric=human_metric)
    print('Random: clu {}, acc {}'.format(clu_rnd, acc_rnd))
    print('Method: clu {}, acc {}'.format(clu_method, acc_method))
    print('='*100)

# Running these datasets get error because of missing datapoints, e.g., some samples miss some system outputs or annotations

# test_custom(data_file='other_data/e2e_challenge.jsonl', method='diversity_bleu', method_metric=None, human_metric='quality')
# test_custom(data_file='other_data/e2e_challenge.jsonl', method='diversity_bleu', method_metric=None, human_metric='naturalness')
# test_custom(data_file='other_data/emnlp_2017.jsonl', method='diversity_bleu', method_metric=None, human_metric='quality')
# test_custom(data_file='other_data/emnlp_2017.jsonl', method='diversity_bleu', method_metric=None, human_metric='naturalness')
# test_custom(data_file='other_data/emnlp_2017.jsonl', method='diversity_bleu', method_metric=None, human_metric='informativeness')


test_custom(data_file='other_data/story_gen_roc.jsonl', method='diversity_bleu', method_metric=None, human_metric='overall')
test_custom(data_file='other_data/story_gen_wc.jsonl', method='diversity_bleu', method_metric=None, human_metric='overall')
test_custom(data_file='other_data/persona_chat.jsonl', method='diversity_bleu', method_metric=None, human_metric='Overall')

test_custom(data_file='other_data/sum_cnndm1_openai.jsonl', method='diversity_bleu', method_metric=None, human_metric='overall')   # human_metric: overall, accuracy, coverage, coherence
test_custom(data_file='other_data/sum_cnndm3_openai.jsonl', method='diversity_bleu', method_metric=None, human_metric='overall')
test_custom(data_file='other_data/sum_dialsum.jsonl', method='diversity_bleu', method_metric=None, human_metric='coherence')
test_custom(data_file='other_data/sum_newsroom.jsonl', method='diversity_bleu', method_metric=None, human_metric='coherence')
