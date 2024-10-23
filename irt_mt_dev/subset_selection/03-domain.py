import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import numpy as np
import tqdm
import random
import collections
import copy

data_old = utils.load_data()
data_domain = collections.defaultdict(list)
domains = set()
for line in data_old:
    domains.add(line["domain"])
    data_domain[line["domain"]].append(line)
domains = list(domains)
points_x = []
points_y = []

for prop in tqdm.tqdm(utils.PROPS):
    k = int(len(data_old) * prop)
    points_x.append(prop)

    points_y_local = []
    # repeat each sampling 10 times to smooth it out
    for _ in range(10):
        data_domain_local = copy.deepcopy(data_domain)
        domain_freq_old = collections.Counter([line["domain"] for line in data_old])
        domain_freq_old = {domain: freq/len(data_old) for domain, freq in domain_freq_old.items()}

        data_new = []
        while len(data_new) < k:
            domain_freq_new = collections.Counter({domain: 0 for domain in domains})
            for line in data_new:
                domain_freq_new[line["domain"]] += 1
            domain_freq_new = {domain: 0 if freq == 0 else freq/len(data_new) for domain, freq in domain_freq_new.items()}

            # go through domains in random order
            random.shuffle(domains)
            # make sure that at least one domain gets through
            domain_freq_new[domains[-1]] -= 1e-10
            for domain in domains:
                # find underrepresented domains
                if domain_freq_new[domain] < domain_freq_old[domain]:
                    el_i = random.randint(0, len(data_domain_local[domain])-1)
                    data_new.append(data_domain_local[domain].pop(el_i))
                    break

        # repeat each sampling 10 times to smooth it out
        points_y_local.append(utils.eval_system_clusters(data_new))

    points_y.append(np.average(points_y_local))

print(f"Average  {np.average(points_y):.2f}")
irt_mt_dev.utils.fig.plot_subset_selection([(points_x, points_y, f"{np.average(points_y):.2f}")], "domain")