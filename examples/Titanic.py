import pathlib
import machine_learning as ml
import machine_learning.decision_trees as dt
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("bmh")

data_path = pathlib.Path(__file__).parents[1] / "data"

df = pd.read_csv(data_path / "Titanic.csv")
y = "Survived"
x = ["Age", "Embarked", "Fare", "Parch", "Pclass", "Sex", "SibSp"]

x_train, y_train, x_test, y_test = ml.train_test(df[x], df[y], 0.1)

model_1 = dt.GBCT()
model_1.fit(x_train, y_train, shrinkage=0.1, min_leaf_samples=100,
            max_leafs_count=30)

model_2 = dt.Classifier()
model_2.fit(x_train, y_train, max_leafs_count=100, min_leaf_samples=100)

model_1.plot_confusion_matrix(x_test, y_test)
model_2.plot_confusion_matrix(x_test, y_test)
plt.show()
