# %%

import numpy as np
import subset2evaluate
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import subset2evaluate.utils

data_old = subset2evaluate.utils.load_data("../../subset2evaluate-tmp/data_other/sum_newsroom.jsonl")
print("Before sanitization", len(data_old))
data_old = subset2evaluate.utils.sanitize_data(data_old, top_systems=2)
print("After sanitization", len(data_old))
print("Metrics", list(data_old[0]["scores"].values())[0].keys())
print("Models", list(data_old[0]["scores"].keys()))

# %%
_ = subset2evaluate.evaluate.eval_metrics_correlations(data_old, metric_target="coherence", display=True)

# %%


def benchmark(data_old, repetitions=1, metric_target=None, method_kwargs={}):
    clu_all = []
    cor_all = []
    for _ in range(repetitions):
        data_new = subset2evaluate.select_subset.basic(
            data_old,
            **method_kwargs,
        )
        clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, data_old, metric=metric_target)
        clu_all.append(clu_new)
        cor_all.append(cor_new)
    print(f'{method_kwargs["method"]:>20}', f"COR: {np.average(cor_all):.1%} | CLU: {np.average(clu_all):.2f}")


# %%
data_old = subset2evaluate.utils.load_data("../../subset2evaluate-tmp/data_other/sum_newsroom.jsonl")
data_old = subset2evaluate.utils.sanitize_data(data_old, top_systems=2)
benchmark(data_old, repetitions=100, metric_target="coherence", method_kwargs={"method": "random"})
benchmark(data_old, repetitions=1, metric_target="coherence", method_kwargs={"method": "metric_var", "metric": "bart_score_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="coherence", method_kwargs={"method": "metric_avg", "metric": "bart_score_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="coherence", method_kwargs={"method": "metric_cons", "metric": "bart_score_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="coherence", method_kwargs={"method": "diversity", "metric": "chrf"})
benchmark(data_old, repetitions=1, metric_target="coherence", method_kwargs={"method": "diversity", "metric": "unigram"})

# %%
data_old = subset2evaluate.utils.load_data("../../subset2evaluate-tmp/data_other/sum_dialsum.jsonl")
data_old = subset2evaluate.utils.sanitize_data(data_old, top_systems=2)
benchmark(data_old, repetitions=100, metric_target="relevance", method_kwargs={"method": "random"})
benchmark(data_old, repetitions=1, metric_target="relevance", method_kwargs={"method": "metric_var", "metric": "bart_score_cnn_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="relevance", method_kwargs={"method": "metric_avg", "metric": "bart_score_cnn_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="relevance", method_kwargs={"method": "metric_cons", "metric": "bart_score_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="relevance", method_kwargs={"method": "diversity", "metric": "chrf"})
benchmark(data_old, repetitions=1, metric_target="relevance", method_kwargs={"method": "diversity", "metric": "unigram"})

# %%
data_old = subset2evaluate.utils.load_data("../../subset2evaluate-tmp/data_other/sum_cnndm3_openai.jsonl")
data_old = subset2evaluate.utils.sanitize_data(data_old, top_systems=2)
benchmark(data_old, repetitions=100, metric_target="overall", method_kwargs={"method": "random"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "metric_var", "metric": "bart_score_cnn_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "metric_avg", "metric": "bart_score_cnn_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "metric_cons", "metric": "bart_score_cnn_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "diversity", "metric": "chrf"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "diversity", "metric": "unigram"})

# %%
data_old = subset2evaluate.utils.load_data("../../subset2evaluate-tmp/data_other/sum_cnndm1_openai.jsonl")
data_old = subset2evaluate.utils.sanitize_data(data_old, top_systems=2)
benchmark(data_old, repetitions=100, metric_target="overall", method_kwargs={"method": "random"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "metric_var", "metric": "bart_score_cnn_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "metric_avg", "metric": "bart_score_cnn_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "metric_cons", "metric": "bart_score_cnn_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "diversity", "metric": "chrf"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "diversity", "metric": "unigram"})

# %%
data_old = subset2evaluate.utils.load_data("../../subset2evaluate-tmp/data_other/story_gen_wc.jsonl")
data_old = subset2evaluate.utils.sanitize_data(data_old, top_systems=2)
benchmark(data_old, repetitions=100, metric_target="overall", method_kwargs={"method": "random"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "metric_var", "metric": "bart_score_cnn_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "metric_avg", "metric": "bart_score_cnn_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "metric_cons", "metric": "bart_score_cnn_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "diversity", "metric": "chrf"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "diversity", "metric": "unigram"})

# %%
data_old = subset2evaluate.utils.load_data("../../subset2evaluate-tmp/data_other/story_gen_roc.jsonl")
data_old = subset2evaluate.utils.sanitize_data(data_old, top_systems=2)
benchmark(data_old, repetitions=100, metric_target="overall", method_kwargs={"method": "random"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "metric_var", "metric": "bart_score_cnn_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "metric_avg", "metric": "bart_score_cnn_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "metric_cons", "metric": "bart_score_cnn_src_hypo"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "diversity", "metric": "chrf"})
benchmark(data_old, repetitions=1, metric_target="overall", method_kwargs={"method": "diversity", "metric": "unigram"})

# %%
data_old = subset2evaluate.utils.load_data("../../subset2evaluate-tmp/data_other/persona_chat.jsonl")
data_old = subset2evaluate.utils.sanitize_data(data_old, top_systems=2)
benchmark(data_old, repetitions=100, metric_target="Overall", method_kwargs={"method": "random"})
benchmark(data_old, repetitions=1, metric_target="Overall", method_kwargs={"method": "metric_var", "metric": "bert_score_r"})
benchmark(data_old, repetitions=1, metric_target="Overall", method_kwargs={"method": "metric_avg", "metric": "bert_score_r"})
benchmark(data_old, repetitions=1, metric_target="Overall", method_kwargs={"method": "metric_cons", "metric": "bert_score_r"})
benchmark(data_old, repetitions=1, metric_target="Overall", method_kwargs={"method": "diversity", "metric": "chrf"})

# %%
data_old = subset2evaluate.utils.load_data("../../subset2evaluate-tmp/data_other/emnlp_2017.jsonl")
data_old = subset2evaluate.utils.sanitize_data(data_old, top_systems=2)
benchmark(data_old, repetitions=1, method_kwargs={"method": "diversity", "metric": "chrf"})
benchmark(data_old, repetitions=1, method_kwargs={"method": "diversity", "metric": "BLEU"})
benchmark(data_old, repetitions=1, method_kwargs={"method": "metric_var", "metric": "ROUGE_L"})
benchmark(data_old, repetitions=1, method_kwargs={"method": "metric_avg", "metric": "ROUGE_L"})
benchmark(data_old, repetitions=1, method_kwargs={"method": "metric_cons", "metric": "ROUGE_L"})

# %%
data_old = subset2evaluate.utils.load_data("../../subset2evaluate-tmp/data_other/e2e_challenge.jsonl")
data_old = subset2evaluate.utils.sanitize_data(data_old, top_systems=2)
benchmark(data_old, repetitions=100, method_kwargs={"method": "random"})
benchmark(data_old, repetitions=1, method_kwargs={"method": "diversity", "metric": "BLEU"})
