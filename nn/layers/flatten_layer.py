from .layer import Layer


class FlattenLayer(Layer):
    def __init__(self, parent=None):
        super(FlattenLayer, self).__init__(parent)

    def forward(self, data):
        # TODO reshape the data here and return it (this can be in place).
        self.x = data
        self.batch_size = self.x.shape[0]
        self.in_c = self.x.shape[1]      # input channel count
        self.in_h = self.x.shape[2]      # input image height
        self.in_w = self.x.shape[3]      # input image width
        return self.x.reshape(self.batch_size,-1)

    def backward(self, previous_partial_gradient):
        # TODO
        reshape_z = previous_partial_gradient.reshape(self.batch_size, self.in_c, self.in_h, self.in_w)
        return reshape_z
