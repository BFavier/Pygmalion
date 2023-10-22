import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygmalion.neural_networks import ProbabilityDistribution


model = ProbabilityDistribution(inputs=["x"], hidden_features=[10], normalize=False)
df = pd.DataFrame(data=np.linspace(0., 1., 100)[:, None], columns=["x"])
f, ax  = plt.subplots()
ax.plot(list(range(len(df))), model.pdf(df), label="analytical")
ax.plot(list(range(len(df) - 1)), np.diff(model.cdf(df)) / np.diff(df.x), label="numerical")
f.legend()
plt.show()


if __name__ == "__main__":
    import IPython
    IPython.embed()
