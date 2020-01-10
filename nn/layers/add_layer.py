from collections.abc import Iterable
from typing import Tuple

import numpy as np

from .layer import Layer


class AddLayer(Layer):
    def __init__(self, parents = None):
        super(AddLayer, self).__init__(parents)

    def forward(self, inputs: Iterable):
        # TODO: Add all the items in inputs. Hint, python's sum() function may be of use.
        self.shape = len(inputs)
        inputs_Sum = sum(inputs)
        #print('sum',len(inputs))
        return inputs_Sum

    def backward(self, previous_partial_gradient) -> Tuple[np.ndarray, ...]:
        # TODO: You should return as many gradients as there were inputs.
        #   So for adding two tensors, you should return two gradient tensors corresponding to the
        #   order they were in the input.
        # back = ()
        # for i in range(self.shape):
        #     back += previous_partial_gradient
        back = (previous_partial_gradient,previous_partial_gradient)
        return back
