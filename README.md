# Pygmalion

Pygmalion in the greek mythologie is a sculptor that fell in love with one of his creations.
In the myth, Aphrodite gives life to Galatea, the sculpture he fell in love with.

This package is a machine learning library. It contains all the tools you need to give a mind of their own to inanimate computers.

## installing pygmalion

pygmalion can be installed through pip.

~~~
pip install pygmalion
~~~

## Fast prototyping with pygmalion

Architectures for several common machine learning tasks (regression, image classification, ...) are implemented in this package.

The inputs and outputs of the model are common python object (such as numpy array and pandas dataframes) so that the learning curve is not too steep.

In this part we are going to see how to load a dataset, train a model, and display some metrics. As a first step you can import the following packages.

~~~python
>>> import pygmalion as ml
>>> import pygmalion.neural_networks as nn
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
~~~

You can download a dataset and split it with the **split** function.

~~~python
>>> ml.datasets.boston_housing("./")
>>> df = pd.read_csv("./boston_housing.csv")
>>> x, y = df[[c for c in df.columns if c != "medv"]], df["medv"]
>>> data, test_data = ml.split((x, y), frac=0.1)
>>> train_data, val_data = ml.split(data, frac=0.1)
~~~

Creating and training a model is done in a few lines of code.

~~~python
>>> hidden_layers = [{"channels": 8}, {"channels": 8}]
>>> model = nn.DenseRegressor(x.columns, hidden_layers, learning_rate=1.0E-3, GPU=False)
>>> model.train(train_data, val_data, n_epochs=1000, patience=100)
~~~

Some usefull metrics can easily be evaluated.

For a regressor model, the available metrics are [**MSE**](https://en.wikipedia.org/wiki/Mean_squared_error), [**RMSE**](https://en.wikipedia.org/wiki/Root-mean-square_deviation), [**R²**](https://en.wikipedia.org/wiki/Coefficient_of_determination), and the correlation between target and prediction can be visualized with the **plot_correlation** function.

~~~python
>>> f, ax = plt.subplots()
>>> x_train, y_train = train_data
>>> ml.plot_correlation(model(x_train), y_train, ax=ax, label="training")
>>> x_val, y_val = val_data
>>> ml.plot_correlation(model(x_val), y_val, ax=ax, label="validation")
>>> x_test, y_test = test_data
>>> ml.plot_correlation(model(x_test), y_test, ax=ax, label="testing", color="C3")
>>> R2 = ml.R2(model(x_test), y_test)
>>> ax.set_title(f"R²={R2:.3g}")
>>> plt.show()
~~~

![pairplot](images/boston_housing_pairplot.png)


For a classifier model you can evaluate the [**accuracy**](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification), and display the confusion matrix.

~~~python
>>> ml.datasets.iris("./")
>>> df = pd.read_csv("./iris.csv")
>>> x, y = df[[c for c in df.columns if c != "variety"]], df["variety"]
>>> inputs, classes = x.columns(), y.unique()
>>> hidden_layers = [{"channels": 5},
>>>                  {"channels": 5},
>>>                  {"channels": 5}]
>>> model = nn.DenseClassifier(inputs, classes,
>>>                            hidden_layers=hidden_layers, >>>                            activation="elu")
>>> data, test_data = ml.split((x, y), frac=0.2)
>>> train_data, val_data = ml.split(data, frac=0.1)
>>> model.train(train_data, val_data, n_epochs=1000, patience=100)
>>> f, ax = plt.subplots()
>>> x_test, y_test = test_data
>>> ml.plot_confusion_matrix(model(x), y, ax=ax)
>>> acc = ml.accuracy(y_pred, y)*100
>>> ax.set_title(f"Accuracy: {acc:.2f}%")
>>> plt.tight_layout()
>>> plt.show()
~~~

![confusion matrix](images/iris_confusion_matrix.png)

All the models can be dumped as a dictionnary through the **dump** property. A copy of the model can be loaded with the **from_dump** class method.

~~~python
>>> dump = model.dump
>>> model = nn.DenseRegressor.from_dump(dump)
~~~

The models can also be be saved directly to the disk in json format with the **save** method.
A model saved on the disk can then be loaded back with the **load** class method.

~~~python
>>> model.save("./model.json")
>>> model = nn.DenseRegressor.load("./model.json")
~~~

## Implemented models

### Neural networks

The neural network models all share some common attributes:

* The **module** attribute is the underlying pytorch Module object.
* The **GPU** attribute is a boolean defining if the model must be evaluated on CPU or GPU.
* The **norm_update_factor** attribute is the factor used to update the batch normalization running mean and variance.
* The **learning_rate** attribute is the learning rate used during training.
* The **optimization_method** attributes is the string name of the torch.optim optimizer used during training.
* The **L1**/**L2** attribute is the L1/L2 penalization factor used during training.
* The **residuals** attribute is a dict containing the training and validation loss.

The **train** method is used to train the neural networks model. The prototype of the method is: 

~~~python
def train(self, training_data: Union[tuple, Callable],
          validation_data: Union[tuple, Callable, None] = None,
          n_epochs: int = 1000,
          patience: int = 100,
          verbose: bool = True,
          L_minibatchs: Union[int, None] = None):
~~~

The parameter **training_data**, which must be a tuple of (x, y) or (x, y, weight) for weighted observations. The types of x/y/weights depends on the model types. It can also be a function that yields the data. This is usefull if the data don't fit all at once in the memory, in which case the 

The history of the loss can be plotted using the **plot_residuals** method.

~~~python
>>> model.train(training_data, validation_data=validation_data,
>>>             n_epochs=1000, patience=100)
>>> model.plot_residuals()
>>> plt.show()
~~~

![residuals](images/boston_housing_residuals.png)

The black line represents the epoch for which the validation loss was the lowest. At each epoch the state of the model is saved if the validation loss has improved. The training stops when the model trained for **n_epochs**, or when the validation loss has not improved for **patience** epochs. After what the last saved state is loaded.

1. **DenseRegressor**

A dense regressor (or multi layer perceptron regressor) predicts a scalar value given an input of several variables.
This implementation takes in input a dataframe of numerical observations, and predict a numpy array of the same length.

It is implemented as a sucession of hidden **Activated0d** layers (linear weighting/non linear activation/batch normalization) and a final linear weighting.

2. **DenseClassifier**

A dense classifier is a 

3. **ImageClassifier**

4. **SemanticSegmenter**



