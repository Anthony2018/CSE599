from numba import njit, prange

from .layer import Layer


class LeakyReLULayer(Layer):
    def __init__(self, slope: float = 0.1, parent=None):
        super(LeakyReLULayer, self).__init__(parent)
        self.slope = slope

    def forward(self, data):
        # TODO
        self.input_data = data
        # data = ((data <= 0) * self.slope * data)
        # data = ((data > 0) * data)
        y1 = ((data > 0) * data)                                                 
        y2 = ((data <= 0) * data * self.slope)                                         
        data = y1 + y2 
        return data

    def backward(self, previous_partial_gradient):
        # TODO
        y1 = (1. * (self.input_data > 0))                                                
        y2 = ((self.input_data <= 0) * self.slope) 
        data_output = y1 + y2
        previous_partial_gradient = previous_partial_gradient * data_output
        return previous_partial_gradient
