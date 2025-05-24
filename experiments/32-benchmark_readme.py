# %%
import subset2evaluate
import subset2evaluate.evaluate
import tqdm
import numpy as np
import json


def benchmark_method(repetitions, data, target="human", kwargs_dict={}):
    if data == "wmt/all":
        data_old_all = list(subset2evaluate.utils.load_data_wmt_all(normalize=True).values())[:9]
    elif data == "summeval":
        data_old_all = [subset2evaluate.utils.load_data(data)]
    points_y_cor = []
    points_y_clu = []

    for data_old in tqdm.tqdm(data_old_all):
        # run multiple times to smooth variance
        for _ in range(repetitions):
            clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(
                subset2evaluate.select_subset.basic(data_old, **kwargs_dict),
                data_old,
                metric=target,
            )
            points_y_cor.append(cor_new)
            points_y_clu.append(clu_new)

    print(f"{data:>10} {json.dumps(kwargs_dict, ensure_ascii=False)} | {np.average(points_y_cor):.1%} | {np.average(points_y_clu):.2f} |")


def benchmark_method_mt(**kwargs):
    benchmark_method(data="wmt/all", target="human", **kwargs)


def benchmark_method_summeval(**kwargs):
    benchmark_method(data="summeval", target="human_sum", **kwargs)


# %%
benchmark_method_mt(repetitions=100, kwargs_dict=dict(method="random"))

# %%
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="metric_var", metric="MetricX-23"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="metric_avg", metric="MetricX-23"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="metric_cons", metric="MetricX-23"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="diversity", metric="BLEU"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="diversity", metric="unigram"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="diversity", metric="lm"))
benchmark_method_mt(repetitions=5, kwargs_dict=dict(method="pyirt_diffdisc", metric="MetricX-23", retry_on_error=True))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="precomet_diversity"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="precomet_avg"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="precomet_var"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="precomet_diffdisc_direct"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="precomet_cons"))

# %%
benchmark_method_summeval(repetitions=100, kwargs_dict=dict(method="random"))
benchmark_method_summeval(repetitions=1, kwargs_dict=dict(method="metric_var", metric="coverage"))
benchmark_method_summeval(repetitions=1, kwargs_dict=dict(method="metric_avg", metric="coverage"))
benchmark_method_summeval(repetitions=1, kwargs_dict=dict(method="metric_cons", metric="coverage"))
benchmark_method_summeval(repetitions=5, kwargs_dict=dict(method="pyirt_diffdisc", metric="coverage"))
benchmark_method_summeval(repetitions=1, kwargs_dict=dict(method="diversity", metric="BLEU"))
benchmark_method_summeval(repetitions=1, kwargs_dict=dict(method="diversity", metric="unigram"))
benchmark_method_summeval(repetitions=1, kwargs_dict=dict(method="diversity", metric="ChrF"))
benchmark_method_summeval(repetitions=1, kwargs_dict=dict(method="diversity", metric="lm"))
