# import machine_learning.neural_networks as neural_networks
# import machine_learning.decision_trees as decision_trees
# import machine_learning.agnostic as agnostic
# import machine_learning.library as library
# from ._linear import *
# from ._custom_regressor import *
# from ._image import *
# from ._utilities import *
# from machine_learning._templates import load as load
# __version__ = agnostic.__version__

from . import neural_networks
from .utilities import split, kfold, MSE, RMSE, R2, accuracy, plot_correlation, plot_confusion_matrix
__version__ = "0.0.0"
