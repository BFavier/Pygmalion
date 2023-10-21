import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygmalion as ml

DEVICE = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"


def pdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = ((x-0.3)**2 + y**2)**0.5
    return np.exp(-20*(d - 0.5)**2)


X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
DF = pd.DataFrame(data=np.stack([X.reshape(-1), Y.reshape(-1)], axis=1), columns=["x", "y"])
CDF = np.cumsum(np.cumsum(pdf(X, Y), axis=1), axis=0)

x, y = (np.random.uniform(-2, 2, size=5000) for _ in range(2))
df = pd.DataFrame(data=np.stack([x, y], axis=1), columns=["x", "y"])
p = pdf(x, y)
t = np.random.uniform(0., 1., size=(p.size,))
df = df[t < p.reshape(-1)]

model = ml.neural_networks.ProbabilityDistribution(["x", "y"], [50, 50])
model.to(DEVICE)
train_data = model.data_to_tensor(df)
train_losses, val_losses, grad, best_step = model.fit(train_data, validation_data=None, n_steps=5_000, learning_rate=1.0E-3, keep_best=False)
ml.utilities.plot_losses(train_losses, val_losses, grad, best_step, log_scale=True)


ncols = 5
f, axes  = plt.subplots(figsize=[ncols*4+1, 4], ncols=ncols)
axes[0].imshow(pdf(X, Y), extent=[-1, 1, -1, 1], origin="lower", aspect="equal", cmap="coolwarm", interpolation="bilinear", vmin=-1, vmax=1)
axes[0].set_title("Target distribution")
axes[1].scatter(df.x, df.y, color="k", marker=".")
axes[1].set_title("Sampled distribution")
axes[1].set_aspect("equal")
_pdf = model.pdf(DF).reshape(X.shape)
axes[2].imshow(_pdf, extent=[-1, 1, -1, 1], origin="lower", aspect="equal", cmap="coolwarm", interpolation="bilinear", vmin=-_pdf.max(), vmax=_pdf.max())
axes[2].set_title("Learned distribution")
axes[2].set_aspect("equal")
_cdf = model.cdf(DF).reshape(X.shape)
axes[3].imshow(_cdf, extent=[-1, 1, -1, 1], origin="lower", aspect="equal", cmap="viridis", interpolation="bilinear")
axes[3].contour(X, Y, _cdf, colors="w", levels=30)
axes[3].set_title("Learned cumulative distribution")
axes[3].set_aspect("equal")
axes[4].imshow(CDF, extent=[-1, 1, -1, 1], origin="lower", aspect="equal", cmap="viridis", interpolation="bilinear")
axes[4].contour(X, Y, CDF, colors="w", levels=30)
axes[4].set_title("Target cumulative distribution")
axes[4].set_aspect("equal")
for ax in axes:
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
f.tight_layout()
plt.show()


if __name__ == "__main__":
    import IPython
    IPython.embed()
