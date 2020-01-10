import numpy as np
from numba import njit, prange

from nn import Parameter
from .layer import Layer


class PReLULayer(Layer):
    def __init__(self, size: int, initial_slope: float = 0.1, parent=None):
        super(PReLULayer, self).__init__(parent)
        self.slope = Parameter(np.full(size, initial_slope))
        self.momentum = 0.9
        self.learning_rate = 0.01
        self.dleta_slope = np.full(size, 0)
        self.input_data = 0
        self.counter = 0


    def forward(self, data):
        # TODO
        self.input_data = data
        if self.slope.data.size != 1:
            for item in range(data.shape[1]):
                y1 = ((data[:,item,...] > 0) * data[:,item,...])                                                 
                y2 = ((data[:,item,...] <= 0) * data[:,item,...] * self.slope.data[item])                                         
                data[:,item,...] = y1 + y2 
        else:
            y1 = ((data > 0) * data)                                                 
            y2 = ((data <= 0) * data * self.slope.data[0])                                         
            data = y1 + y2 
        return data

    def backward(self, previous_partial_gradient):
        # TODO
        # epsilon=self.slope
        # gradients = 1. * (previous_partial_gradient > epsilon)
        # gradients = (gradients == 0) * 0
        if self.slope.data.size != 1:
            for item in range(previous_partial_gradient.shape[1]):
                para =  ((self.input_data[:,item,...] <= 0) * self.input_data[:,item,...])
                self.slope.grad[item] = np.sum(para * (previous_partial_gradient[:,item,...]))
            #update the slope parameter:
            for i in range(self.input_data.shape[1]):
                y1 = ((self.input_data[:,i,...] > 0) * 1)                                                 
                y2 = ((self.input_data[:,i,...] <= 0) * self.slope.data[i])                                         
                output_data = y1 + y2
                previous_partial_gradient[:,i,...] = previous_partial_gradient[:,i,...] * output_data

            for i in range(previous_partial_gradient.shape[1]):
                self.dleta_slope[i] = self.momentum * self.dleta_slope[i] + self.learning_rate * self.slope.grad[i]
                self.slope.data[i] -= self.dleta_slope[i] 
        else:
            para = ((self.input_data <= 0) * self.input_data)
            # print(temp[:,2,...])
            self.slope.grad[0] = np.sum(para * (previous_partial_gradient))
            y1 = (1 * (self.input_data > 0))                                                
            y2 = ((self.input_data <= 0) * self.slope.data[0]) 
            data_output = y1 + y2
            self.dleta_slope[0] = self.momentum * self.dleta_slope[0] + self.learning_rate * self.slope.grad[0]
            self.slope.data[0] -= self.learning_rate * self.dleta_slope[0] 
            previous_partial_gradient = previous_partial_gradient * data_output
        if self.counter % 100 == 0:
            print(max(self.slope.data),min(self.slope.data))
        self.counter += 1
        return previous_partial_gradient 