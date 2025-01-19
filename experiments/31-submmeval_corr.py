# %%

import subset2evaluate.utils as utils
import scipy.stats
import collections

data_old = utils.load_data_summeval(normalize=True)

# %%
corrs = collections.defaultdict(dict)
models = list(data_old[0]["scores"].keys())
for metric in data_old[0]["scores"]["M11"].keys():
    print(f"{metric:>30}", end=" ")
    for target in ["human_relevance", "human_coherence", "human_consistency", "human_fluency", "human_avg", "human_mul"]:
        data_y = [line["scores"][model][target] for line in data_old for model in models]
        data_x = [line["scores"][model][metric] for line in data_old for model in models]
        corr, _ = scipy.stats.pearsonr(data_x, data_y)
        print(f"{corr:>5.0%}", end=" ")
        corrs[target][metric] = corr
    print()

# %%

# top 10 for each target
for target in ["human_relevance", "human_coherence", "human_consistency", "human_fluency", "human_avg", "human_mul"]:
    print(target)
    for metric, corr in sorted(corrs[target].items(), key=lambda x: -x[1])[:10]:
        print(f"{metric:>20}", f"{corr:>5.0%}")
    print()
