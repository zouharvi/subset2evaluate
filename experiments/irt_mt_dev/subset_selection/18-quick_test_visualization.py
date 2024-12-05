# %% 
import pyro
import pyro.distributions
import matplotlib.pyplot as plt

dist = pyro.distributions.Categorical(0.3)

y = [dist.sample() for _ in range(10_000)]

plt.hist(y, density=True)
plt.show()
# %%
