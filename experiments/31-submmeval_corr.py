# %%

import subset2evaluate.utils as utils
import scipy.stats

data_old = utils.load_data_summeval(normalize=True)

# %%

import collections
corrs = collections.defaultdict(dict)
systems = list(data_old[0]["scores"].keys())
for metric in data_old[0]["scores"]["M11"].keys():
    print(f"{metric:>30}", end=" ")
    for target in ["human_relevance", "human_coherence", "human_consistency", "human_fluency"]:
        data_y = [line["scores"][sys][target] for line in data_old for sys in systems]
        data_x = [line["scores"][sys][metric] for line in data_old for sys in systems]
        corr, _ = scipy.stats.pearsonr(data_x, data_y)
        print(f"{corr:>5.0%}", end=" ")
        corrs[target][metric] = corr
    print()

# %%

# top 10 for each target
for target in ["human_relevance", "human_coherence", "human_consistency", "human_fluency"]:
    print(target)
    for metric, corr in sorted(corrs[target].items(), key=lambda x: -x[1])[:10]:
        print(f"{metric:>20}", f"{corr:>5.0%}")
    print()
