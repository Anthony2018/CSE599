from .. import *


class ResNetBlock(LayerUsingLayer):
    def __init__(self, conv_params, parent=None):
        super(ResNetBlock, self).__init__(parent)
        self.conv_layers = SequentialLayer([ConvLayer(*conv_params), ReLULayer(), ConvLayer(*conv_params)], self.parent)
        parent = self.parent
        # the following for loop is from the layer use layer to inital
        for ll, layer in enumerate(self.conv_layers):
            setattr(self, str(ll), layer)
            layer.parent = parent
            if isinstance(layer, LayerUsingLayer):
                parent = layer.final_layer
            else:
                parent = layer
        self.add_layer = AddLayer(parents = (parent,self.parent))
        self.relu2 = ReLULayer(parent = self.add_layer)
        assert not any([parent is None for parent in self.conv_layers.parents])
        assert not any([parent is None for parent in self.add_layer.parents])
        assert not any([parent is None for parent in self.relu2.parents])

    @property
    def final_layer(self):
        # return self.relu2
        final_layer = self.relu2
        if isinstance(final_layer, LayerUsingLayer):
            return final_layer.final_layer
        return final_layer

    def forward(self, data):
        f_x = data
        for layer in self.conv_layers:
            f_x = layer.forward(f_x)
        after_add = self.add_layer((data,f_x))
        after_relu2 = self.relu2(after_add)
        return after_relu2
