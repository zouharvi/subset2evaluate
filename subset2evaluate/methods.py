import numpy as np

def random(data, args):
    import random
    random.Random(0).shuffle(data)
    return data

def metric_avg(data, args):
    data.sort(key=lambda item: np.average(
        [sys_v[args.metric] for sys_v in item["scores"].values()]
    ))
    return data

def metric_var(data, args):
    data.sort(key=lambda item: np.var(
        [sys_v[args.metric] for sys_v in item["scores"].values()]
    ), reverse=True)
    return data

def irt(data, args):
    pass


METHODS = {
    "random": random,
    "avg": metric_avg,
    "var": metric_var,
    "irt": irt
}