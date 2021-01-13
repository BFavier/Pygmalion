import torch as _torch
import machine_learning.neural_networks.layers as _layers
from machine_learning.neural_networks._regressor import Regressor


# Old monotonic method
class Monotonic(Regressor):
    """Penalization for non Monotonic variations"""
    def __init__(self, **kwargs):
        super(Monotonic, self).__init__(**kwargs)
        self.alpha = 1.
        self.sign = self._as_tensor(1.)
        
    def fit(self, *args, alpha=None, sign=None, **kwargs):
        if not alpha is None:
            self.alpha = alpha
        if not sign is None:
            self.sign = self._as_tensor(sign)
        return Regressor.fit(self, *args, **kwargs)
        
    def fit_batch(self, *args, alpha=None, sign=None, **kwargs):
        if not alpha is None:
            self.alpha = alpha
        if not sign is None:
            self.sign = self._as_tensor(sign)
        return Regressor.fit_batch(self, *args, **kwargs)
    
    def _grad(self, tensor):
        """
        Computes the grad of the prediction with respect to the input data
        This function zeros the gradient of the model's parameters as a side effect
        """
        tensor = tensor.clone().detach().requires_grad_()
        with _torch.enable_grad():
            prediction = self(tensor)
            prediction.sum().backward(create_graph=True)
        grad = tensor.grad
        self.optimizer.zero_grad()
        return grad
    
    def _penalization(self, grad):
        """
        Computes a penalization term for negative gradients, so that the 
        model be monotonically increasing
        """
        grad = grad*self.sign
        negative_grads = _torch.clamp_max(grad, 0.)#only keep the negative gradients
        #This is fidel to the publication
        #normed_grads = _torch.mean(_torch.min(negative_grads, dim=1)[0]**2)
        normed_grads = _torch.mean(_torch.norm(negative_grads, dim=1))
        return self.alpha * normed_grads
    
    def _get_loss_tensor_(self, X, Y):
        """
        compute the loss tensor
        Override the template so that a penalization for negative 
        gradients of the function is added to the loss
        """
        tensor,truth = self._transform_XY(X,Y)
        grad = self._grad(tensor)
        prediction = self(tensor)
        loss = self.loss(prediction,truth) + self._penalization(grad)
        return loss

class Gated(Regressor):
    """The weights of the linear gates are always positive: function always raising"""
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.y_min = None

    @property
    def hidden_layers(self):
        return self._hidden_layers
    
    @hidden_layers.setter
    def hidden_layers(self, other):
        """Generate the hidden layers"""
        if (self._hidden_layers == tuple(other)) and self.fitted:#we dont reset the shape if it is not needed
            return
        self._hidden_layers = tuple(other)
        for i,layer in enumerate(self.layers):
            delattr(self,f"layer{i}")
        self.layers = []
        new_shape = self.shape
        for i,_ in enumerate(new_shape):
            if i == len(new_shape)-1:
                continue
            layer = _layers.LinearPW(new_shape[i],new_shape[i+1])
            #layer = _torch.nn.Linear(new_shape[i],new_shape[i+1])
            setattr(self, f"layer{i}", layer)
            self.layers.append(layer)
        self.reset_optimization()