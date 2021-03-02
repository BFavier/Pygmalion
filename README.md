Pygmalion in the greek mythologie is a sculptor that fell in love with one of his creations.
In the myth, Aphrodite gives life to Galatea, the sculpture he fell in love with. This package is a machine learning library. It contains all the tools you need to give a mind of their own to inanimate objects.

# Installing pygmalion

pygmalion can be installed through pip.

~~~
pip install pygmalion
~~~

# Fast prototyping of models with pygmalion

Architectures for several common machine learning tasks (regression, image classification, ...) are implemented in this package.

The inputs and outputs of the models are common python objects (such as numpy array and pandas dataframes) so there are very few new things you need to learn to use this package.

In this part we are going to see how to load a dataset, train a model, and display some metrics. As a first step you can import the following packages.

~~~python
>>> import pygmalion as ml
>>> import pygmalion.neural_networks as nn
>>> import pandas as pd
>>> import numpy as np
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

![pairplot](https://raw.githubusercontent.com/BFavier/Pygmalion/main/images/boston_housing_pairplot.png)


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
>>>                            hidden_layers=hidden_layers,
>>>                            activation="elu")
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

![confusion matrix](https://raw.githubusercontent.com/BFavier/Pygmalion/main/images/iris_confusion_matrix.png)

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

# Implemented models

For examples of model training see the **samples** folder in the [github page](https://github.com/BFavier/Pygmalion).

## Neural networks

The neural network models all share some common attributes:

* The **GPU** attribute is a boolean defining if the model must be trained/evaluated on GPU. This is only avilable with a CUDA capable GPU. This can massively speed up training/evaluation speed on big models (image analysis, ...).
* The **learning_rate** attribute is the learning rate used during training.
* The **optimization_method** attributes is the string name of the torch.optim optimizer used during training.
* The **L1**/**L2** attribute is the L1/L2 penalization factor used during training.
* The **residuals** attribute is a dict containing the training and validation loss history.
* The **norm_update_factor** attribute is the factor used to update the batch normalization running mean and variance. The default value is mostly always ok, unless you do a lot of batchs/minibatchs where it might benefit to getting reduced.

The classifier neural networks have an additional attribute:

* The **class_weights** parameters is a dict of {class: weight}.

All these attributes are key word arguments of the constructors, and can also be accessed/modified as an attribute of the model after creation.

The neural networks are implemented in pytorch under the hood.
The underlying pytorch Module and Optimizer can be accessed as the **model** and **optimizer** attributes of the model.

The **train** method is used to train the neural networks model. The prototype of the method is: 

~~~python
def train(self, training_data: Union[tuple, Callable],
          validation_data: Union[tuple, Callable, None] = None,
          n_epochs: int = 1000,
          patience: int = 100,
          verbose: bool = True,
          batchs_length: Union[int, None] = None):
~~~

* The parameter **training_data** must be a tuple of (x, y, [weight]). The weight is optional. The types of x/y/weights depends on the model types.

* The parameter **validation_data** is similar to **training_data**. This is the data on which the loss is evaluated, but not back propagated (the model doesn't learn from it). It is used for early stopping: the training stops if the validation loss doesn't improve anymore (to prevent overfitting). This parameter is optional, so that you can verify that your model is able to overfit before trying to train it with early stopping.

* The parameter **n_epochs** is the maximum number of epochs the model performs. Althought the user can still interupt the training early on by pressing ctrl+c (the raised exception is handled and the training simply stops).

* The **patience** parameter is the number of epoch without improvement of the validation loss after which the early stopping triggers.

* The **verbose** parameter describes whether ther train/validation loss shoudl be printed at eahc epoch.

* If the **batchs_length** parameter is not None, the data are shuffled and cut in batches of at most **batchs_length** observations at each epoch. This is necessary to entrain big models on limited GPU memory.

The history of the loss can be plotted using the **plot_residuals** method.

~~~python
>>> model.train(training_data, validation_data=validation_data,
>>>             n_epochs=1000, patience=100)
>>> model.plot_residuals()
>>> plt.show()
~~~

![residuals](https://raw.githubusercontent.com/BFavier/Pygmalion/main/images/boston_housing_residuals.png)

The black line represents the epoch for which the validation loss was the lowest. At each epoch the state of the model is checkpointed if the validation loss has improved. And at the end of the training the last best state is loaded back.

If you restart training by calling **train** a second time, it will starts back from the black line. Model dumped to a dict or saved to the disk are saved with the weights, the optimizer internal parameters, and the gradient. So you can restart your training from where you stopped it without jumps in the loss.

### **DenseRegressor**

A dense regressor (or multi layer perceptron regressor) predicts a scalar value given an input of several variables.

This implementation takes in input **x** a pandas.DataFrame of numerical observations, and returns **y** a numpy.ndarray of floats of the same length. The optional **weights** weighting of the observations during training are numpy.ndarray of floats.

It is implemented as a sucession of hidden **Activated0d** layers (linear weighting/non linear activation/batch normalization) and a final linear weighting to reduces the number of features to one scalar prediction.

The args and kwargs passed to the underlying pytorch Module are:

~~~python
def __init__(self, inputs: List[str],
             hidden_layers: List[dict],
             activation: str = "relu",
             stacked: bool = False,
             dropout: Union[float, None] = None)
~~~

* The **inputs** argument is a list of str, the name of the columns to use as inputs in dataframes
* The **hidden_layers** argument is a list of dict passed as kwargs to the **Activated0d** layers:
    * The **channels** kwarg is the number of features in the layer
    * The **activation** kwarg is the string name of the activation function
    * The **dropout** kwarg (either None or a float between 0 and 1) is dropout rate applied on the channels.
    * The **bias** kwarg is a boolean. If False there is no bias in the linear operation.
    * The **stacked** kwarg is a boolean. If True the features of the input are concatenated to the output of the layer (activation and dropout are not applied to them).
* The **activation** argument is a default value for the **activation** kwargs of the layers
* The **stacked** argument is a default value for the **stacked** kwargs of the layers
* The **dropout** argument is a default value for the **dropout** kwargs of the layers

### **DenseClassifier**

A dense classifier (or multi layer perceptron classifier) predicts a str class value given an input of several variables.

This implementation takes in input **x** a pandas.DataFrame of numerical observations, and returns **y** list of str of the same length. The optional **weights** weighting of the observations during training are numpy.ndarray of floats.

Similarly to the DenseRegressor it is a succession of hidden **Activated0d** layers, and a final linear layer with as much output as there are classes to predict.

The args and kwargs passed to the underlying pytorch Module are mostly the same as the DenseRegressor:

~~~python
>>> def __init__(self, inputs: List[str], classes: List[str],
>>>              hidden_layers: List[dict],
>>>              activation: str = "relu", stacked: bool = False,
>>>              dropout: Union[float, None] = None):
~~~

The only addition is the **classes** argument, which is a list of the unique str classes the model can classify into.

### **ImageClassifier**

An ImageClassifier predicts a str class given as input an image. Here below the predictions of a model trained on the fashion-MNIST dataset. You can see at the 2nd row 3rd column, it missclassified pullover as a hand bag.

![fashion-MNIST predictions](https://raw.githubusercontent.com/BFavier/Pygmalion/main/images/Fashion_MNIST_illustration.png)

It is implemented as a Convolutional Neural Network similar to LeNet.

The args and kwargs passed to the underlying pytorch Module are:

~~~python
>>> def __init__(self, in_channels: int,
>>>              classes: List[str],
>>>              convolutions: Union[List[dict], List[List[dict]]],
>>>              pooling: List[Tuple[int, int]],
>>>              dense: List[dict],
>>>              pooling_type: str = "max",
>>>              padded: bool = True,
>>>              activation: str = "relu",
>>>              stacked: bool = False,
>>>              dropout: Union[float, None] = None):
~~~

* The **in_channels** argument is the number of channels in the input images (1 for grayscale, 3 for RGB, 4 for RGBA, ...)
* The **classes** argument is a list of the unique str classes the model can classify into.
* The **convolutions** argument is a list of list of dict. Each item in the list is a pooling stage. Before pooling a succession of **Activated2d** layers are applied. The dict contains the kwargs passed to the **Activated0d**:
    * The **channels** kwarg is the number of channels as output of the convolution
    * The **window** kwarg is a tuple of (height, width) integers, defining the size of the convolution window
    * The **stride** kwarg is a tuple of (dy, dx) integers, defining the steps performed by the window
    * The **padded** kwarg is a boolean. If True the input feature map is padded to have the same (height, width) after the convolution
    * The **activation** kwarg is the string name of the activation function
    * The **dropout** kwarg (either None or a float between 0 and 1) is the dropout rate applied on the channels.
    * The **bias** kwarg is a boolean. If False there is no bias in the convolution operations.
    * The **stacked** kwarg is a boolean. If True the input feature map is downsampled (if some stride is used) and concatenated to the output of the layer (activation and dropout are not applied to the downsampled input).
* The **pooling** argument is a list of (height, width) tuples, defining the pooling window at eahc pooling stage. The last pooling window is an over-all pooling and so must not be included.
* The **dense** argument is a list of kwargs used to create a **Dense0d** layer (similar to the hidden layers of a DenseRegressor). This is used to further process the single pixel feature maps output of the previous stage.
* The **pooling_type** argument is the type of pooling performed. It must be one of {"max", "avg"} for max pooling and average pooling.
* The **padded** argument is a default value for the **padded** kwargs of the layers
* The **activation** argument is a default value for the **activation** kwargs of the layers
* The **stacked** argument is a default value for the **stacked** kwargs of the layers
* The **dropout** argument is a default value for the **dropout** kwargs of the layers

### **SemanticSegmenter**

![segmented_cityscapes](https://raw.githubusercontent.com/BFavier/Pygmalion/main/images/segmented_cityscape.png)


