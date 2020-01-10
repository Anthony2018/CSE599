from .base_optimizer import BaseOptimizer


class MomentumSGDOptimizer(BaseOptimizer):
    def __init__(self, parameters, learning_rate, momentum=0.9, weight_decay=0):
        super(MomentumSGDOptimizer, self).__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.previous_deltas = [0] * (len(parameters))
        self.counter = 0
        self.num_para = 0
        self.parameter_dic={}

    def step(self):
        self.num_para = 0
        for parameter in self.parameters:
            # TODO update the parameters
            # initial all parameters 
            if self.counter == 0:
                self.previous_w = parameter.grad + self.weight_decay * parameter.data
                self.parameter_dic[self.num_para] = self.previous_w 
            else:
                self.parameter_dic[self.num_para]= self.momentum * self.parameter_dic[self.num_para] + parameter.grad + self.weight_decay * parameter.data
            parameter.data -= self.learning_rate * self.parameter_dic[self.num_para]
            self.num_para = self.num_para + 1
            #print(parameter.data.shape)
            #print(self.counter)
        self.counter = 1
