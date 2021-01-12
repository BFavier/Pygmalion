"""
Mahul, A., & Aussem, A. (2004).
Training feed-forward neural networks with monotonicity requirements.
Research Report RR-04-11, LIMOS/CNRS 6158.
"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import machine_learning.neural_networks as nn
plt.style.use("bmh")
path = pathlib.Path(__file__)


X = np.linspace(-1., 1., 1000)
Y = X**3 - 5*X*np.exp(-100*X**2) + np.random.uniform(-0.05, 0.05, len(X))

df = pd.DataFrame(data=np.transpose([X, Y]), columns=["x", "y"])
df_train = df.sample(frac=0.8, random_state=200)  # random state is a seed value
df_validation = df.drop(df_train.index)
x = ["x"]
y = "y"

kwargs = {"patience": None, "n_epochs": 1000, "hidden_layers": (32,16,8),
          "validation_data":df_validation, "non_linear":"sigmoid", "learning_rate":1.0E-2}

model1 = nn.Regressor()
model1.fit(df_train, x, y, **kwargs)
model1.plot_history(log=True)
Y1 = model1.predict(df)

model2 = nn.Gated()
model2.fit(df_train, x, y, **kwargs)
model2.plot_history(log=True)
Y2 = model2.predict(df)

f,ax = plt.subplots()
ax.scatter(df_train.x, df_train.y, label="training data", color="g", marker=".", s=20.)
ax.scatter(df_validation.x, df_validation.y, label="validation data", color="#EE82EE", marker=".", s=20.)
ax.plot(X, Y1, label="Unconstrained NN", color="r")
ax.plot(X, Y2, label="Monotonic NN", color="b")
ax.legend()
f.tight_layout()

plt.show()
