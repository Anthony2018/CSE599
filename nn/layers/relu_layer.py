import numpy as np
from numba import njit, prange

from .layer import Layer


class ReLULayer(Layer):
    def __init__(self, parent=None):
        super(ReLULayer, self).__init__(parent)

    def forward(self, data):
        # TODO
        self.input_data = data
        return np.maximum(data,0)

    def backward(self, previous_partial_gradient):
        # print(dZ.shape)
        # print(self.input_data.shape)
        #return np.where(dZ > 0, 1.0, 0.0) 
        #relu_grad = self.input_data>0      
        #return previous_partial_gradient*relu_grad
        g_int = np.maximum(self.input_data,0)
        g_int = ((g_int)>0)  
        return previous_partial_gradient * g_int



class ReLUNumbaLayer(Layer):
    def __init__(self, parent=None):
        super(ReLUNumbaLayer, self).__init__(parent)
        self.data = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        # TODO Helper function for computing ReLU
        return np.maximum(data,0)

    def forward(self, data):
        # TODO
        self.data = data
        output = self.forward_numba(data)
        return output

    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, grad):
        # TODO Helper function for computing ReLU gradients
        g_int = np.maximum(data,0)
        g_int = ((g_int)>0) 
        return grad * g_int

    def backward(self, previous_partial_gradient):
        # TODO
        back = self.backward_numba(self.data, previous_partial_gradient)
        return back