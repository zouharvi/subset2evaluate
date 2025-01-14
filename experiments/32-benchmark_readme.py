# %%
import subset2evaluate
import tqdm
import numpy as np
import json


def benchmark_method(repetitions, data, target="human", kwargs_dict={}):
    if data == "wmt/all":
        data_old_all = list(subset2evaluate.utils.load_data_wmt_all(normalize=True).values())[:9]
    elif data == "summeval":
        data_old_all = [subset2evaluate.utils.load_data(data)]
    points_y_acc = []
    points_y_clu = []

    for data_old in tqdm.tqdm(data_old_all):
        # run multiple times to smooth variance
        for _ in range(repetitions):
            clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(
                subset2evaluate.select_subset.run_select_subset(data_old, **kwargs_dict, retry_on_error=False),
                data_old,
                metric=target,
            )
            points_y_acc.append(acc_new)
            points_y_clu.append(clu_new)

    print(f"{data:>10} {json.dumps(kwargs_dict, ensure_ascii=False)} | {np.average(points_y_acc):.1%} | {np.average(points_y_clu):.2f}")


def benchmark_method_mt(**kwargs):
    benchmark_method(data="wmt/all", target="human", **kwargs)


def benchmark_method_summeval(**kwargs):
    benchmark_method(data="summeval", target="human_mul", **kwargs)


# %%
benchmark_method_mt(repetitions=100, kwargs_dict=dict(method="random"))

# %%
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="metric_var", metric="MetricX-23"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="metric_avg", metric="MetricX-23"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="metric_var", metric="bleu"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="metric_avg", metric="bleu"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="diversity_bleu"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="diversity_unigram"))
benchmark_method_mt(repetitions=5, kwargs_dict=dict(method="pyirt_diffdisc", metric="MetricX-23"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="precomet_diversity"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="precomet_diffdisc"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="precomet_diffdisc_direct"))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="precomet_avg", reverse=True))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="precomet_avg", reverse=False))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="precomet_var", reverse=True))
benchmark_method_mt(repetitions=1, kwargs_dict=dict(method="precomet_var", reverse=False))

# %%
benchmark_method_summeval(repetitions=100, kwargs_dict=dict(method="random"))
benchmark_method_summeval(repetitions=1, kwargs_dict=dict(method="metric_var", metric="coverage"))
benchmark_method_summeval(repetitions=1, kwargs_dict=dict(method="metric_avg", metric="coverage"))
benchmark_method_summeval(repetitions=5, kwargs_dict=dict(method="pyirt_diffdisc", metric="coverage"))
benchmark_method_summeval(repetitions=1, kwargs_dict=dict(method="diversity_bleu"))
benchmark_method_summeval(repetitions=1, kwargs_dict=dict(method="diversity_unigram"))
