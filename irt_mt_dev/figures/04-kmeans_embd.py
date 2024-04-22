import utils
import utils.fig
import numpy as np
import tqdm
import random
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2").to("cuda:0")

def vec_sim(a, b):
    # return -np.linalg.norm(a-b)
    return np.dot(a, b)

data_old = utils.load_data()
# encode references
for line in tqdm.tqdm(data_old):
    line["feat"] = model.encode([line["ref"]])[0]

# average translation representations
# for line in tqdm.tqdm(data_old):
#     line["feat"] = np.average(model.encode(list(line["tgt"].values())), axis=0)

points_x = []
points_y = []

for prop in tqdm.tqdm(utils.PROPS):
    points_x.append(prop)

    points_y_local = []
    # repeat each sampling 10 times to smooth it out
    for _ in range(10):
        data_prototypes = random.sample(data_old, k=int(len(data_old) * prop))

        data_new = []
        for line in data_old:
            # find nearest prototypical example
            data_new.append(
                max(
                    data_prototypes, key=lambda x: vec_sim(line["feat"], x["feat"])
                )
            )

        # repeat each sampling 10 times to smooth it out
        points_y_local.append(utils.eval_data_pairs(data_new, data_old))

    points_y.append(np.average(points_y_local))

print(f"Average  {np.average(points_y):.2%}")

utils.fig.plot_subsetacc([(points_x, points_y, f"{np.average(points_y):.2%}")], "kmeans_embd")