import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygmalion as ml

DEVICE = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"


# def pdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     return (np.sin(10 * ((x-0.3)**2 + y**2)**0.5) + 1)/2

# def pdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     return np.exp(-((x-0.3)**2 + (y**2))) + np.exp(-((x+0.3)**2 + ((y+0.1)**2)))


def pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-10*(x+0.6)**2) + 0.6 * np.exp(-10*(x-0.6)**2)


X = np.linspace(-1, 1, 1000)
DF = pd.DataFrame(data=X[:, None], columns=["x"])

x = np.random.uniform(-1, 1, size=1000)
df = pd.DataFrame(data=x[:, None], columns=["x"])
p = pdf(x)
t = np.random.uniform(0., 1., size=(p.size,))
df = df[t < p.reshape(-1)].sort_values("x")

model = ml.neural_networks.ProbabilityDistribution(["x"], [50, 50], normalize=False)
model.to(DEVICE)
train_data = model.data_to_tensor(df)
train_losses, val_losses, grad, best_step = model.fit(train_data, validation_data=None, n_steps=1000, learning_rate=1.0E-3, keep_best=False)
ml.utilities.plot_losses(train_losses, val_losses, grad, best_step, log_scale=True)


ncols = 3
f, axes  = plt.subplots(figsize=[ncols*4+1, 4], ncols=ncols)
axes[0].plot(X, pdf(X))
axes[0].set_title("Target distribution")
axes[1].hist(df.x, bins=50)
axes[1].set_title("Sampled distribution")
axes[2].plot(DF.x, model.pdf(DF))
axes[2].set_title("Learned distribution")
for ax in axes:
    ax.set_xlim([-1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
f.tight_layout()


ncols = 3
f, axes  = plt.subplots(figsize=[ncols*4+1, 4], ncols=ncols)
axes[0].plot(X, np.cumsum(pdf(X)))
axes[0].set_title("Target distribution")
axes[1].step(sorted(df.x), range(len(df)))
axes[1].set_title("Sampled distribution")
axes[2].plot(DF.x, model.cdf(DF))
axes[2].set_title("Learned distribution")
for ax in axes:
    ax.set_xlim([-1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
f.tight_layout()

plt.show()


if __name__ == "__main__":
    import IPython
    IPython.embed()
