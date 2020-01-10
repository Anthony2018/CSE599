import numpy as np

from .loss_layer import LossLayer


class SoftmaxCrossEntropyLossLayer(LossLayer):
    def __init__(self, reduction="mean", parent=None):
        """

        :param reduction: mean reduction indicates the results should be summed and scaled by the size of the input (excluding the axis dimension).
            sum reduction means the results should be summed.
        """
        self.reduction = reduction
        super(SoftmaxCrossEntropyLossLayer, self).__init__(parent)

    def forward(self, logits, targets, axis=-1) -> float:
        """

        :param logits: N-Dimensional non-softmaxed outputs. All dimensions (after removing the "axis" dimension) should have the same length as targets.
            Example: inputs might be (4 x 10), targets (4) and axis 1.
        :param targets: (N-1)-Dimensional class id integers.
        :param axis: Dimension over which to run the Softmax and compare labels.
        :return: single float of the loss.
        """
        # TODO
        self.targets = targets
        self.axis = axis
        maxlog = np.max(logits, axis=axis,keepdims=True)
        #maxlog = maxlog.reshape(maxlog.size,1)
        #print(maxlog.size)
        #print("logist", logits)
        #print("maxa", maxa)
        logits =  logits - maxlog
        self.logits = logits
        #print("logist", logits)
        m=targets.shape[0]
        # p = np.exp(logits)/np.sum(np.exp(logits))
        # #p=p.mean(axis=0)
        # print("softmax",p.shape)
        # log_likelihood = -np.log(p[range(m),targets])
        # print("log fun",log_likelihood.shape)
        log_likelihood = logits - np.log(np.sum(np.exp(logits), axis=axis ,keepdims=True))
        log_likelihood = log_likelihood[range(m),targets]
        if self.reduction == 'mean':
            loss = -(np.sum(log_likelihood))/ m
        else:
            loss = -(np.sum(log_likelihood))
        #print(loss)
        return loss


    def backward(self) -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        # TODO
        m = self.targets.shape[0]
        exps = np.exp(self.logits)
        sum_exps = np.sum(np.exp(self.logits), axis=self.axis ,keepdims=True)
        grad = exps/sum_exps
        # qx = exps/sum_exps
        grad[range(m),self.targets] -= 1
        if self.reduction == 'mean':
             grad = grad/m
        return grad
