from typing import Optional, Callable
import numpy as np
from numba import njit, prange
from nn import Parameter
from .layer import Layer


class ConvLayer(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, parent=None):
        super(ConvLayer, self).__init__(parent)
        self.weight = Parameter(np.zeros((input_channels, output_channels, kernel_size, kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros(output_channels, dtype=np.float32))
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.initialize()

    #method
    # @staticmethod
    # @njit(parallel=True, cache=True)
    # def forward_numba(prepare_data, kernel_h, kernel_w,local_out_h,local_out_w, stride, pre_col):
    #     for row in prange(kernel_h):
    #         row_end = row + stride*local_out_h
    #         for col in prange(kernel_w):
    #             col_end = col + stride*local_out_w
    #             pre_col[:, :, row, col, :, :] = prepare_data[:, :, row:row_end:stride, col:col_end:stride]
    #     return pre_col
    def forward_numba(self, prepare_data, local_kh, local_kw,local_oh,local_ow, stride, pre_col):
        for row in range(local_kh):
            row_end = row + stride*local_oh
            for col in range(local_kw):
                col_end = col + stride*local_ow
                pre_col[:, :, row, col, :, :] = prepare_data[:, :, row:row_end:stride, col:col_end:stride]
        return pre_col

            
    def forward_helper(self, input_data, kernel_h, kernel_w, stride=1, padding=0):
        local_N, local_C, local_H, local_W = input_data.shape
        # padding fisrt
        if padding > 0:
            prepare_data = np.pad(input_data, [(0,0), (0,0), (padding, padding), (padding, padding)], 'constant')
        else:
            prepare_data =input_data
        # calcalate the local szie
        local_out_h = (local_H + 2*padding - kernel_h)//stride + 1
        local_out_w = (local_W + 2*padding - kernel_w)//stride + 1
        # genernate the col with 0
        pre_col = np.zeros((
            local_N, local_C, 
            kernel_h, kernel_w, 
            local_out_h, local_out_w))
        # use numba to do the for loop
        mid_col = self.forward_numba(prepare_data, kernel_h, kernel_w,local_out_h,local_out_w, stride, pre_col)
        # reshpe back
        reshape_size = local_N*local_out_h*local_out_w
        final_col = np.transpose(mid_col, axes=(0, 4, 5, 1, 2, 3)).reshape(reshape_size, -1)
        return final_col

    def forward(self, data):
        # remember the data and different size
        # remember the input_data
        self.input_data = data
        # remember the batch size
        self.batch_size = self.input_data.shape[0]
        # remember the channel
        self.input_channel = self.input_data.shape[1]
        self.output_channel = self.weight.data.shape[1]
        # input size
        self.input_height= self.input_data.shape[2]
        self.input_width = self.input_data.shape[3]
        # output size
        self.kernel_height = self.weight.data.shape[2]
        self.kernel_width= self.weight.data.shape[3] 
        # calclulate the output size by padding.
        self.output_height = (self.input_height - self.kernel_height + 2 * self.padding) // self.stride + 1    
        self.output_width = (self.input_width - self.kernel_width + 2 * self.padding) // self.stride + 1
        #assert(self.input_data.shape == (self.batch_size, self.input_channel, self.input_height, self.input_width))
        #calclulate the data
        self.after_col_data = self.forward_helper(
            self.input_data, self.kernel_height, 
            self.kernel_width, self.stride, self.padding
            )
        # change the weight's axis to fit the calulate:
        new_weigth = np.moveaxis(self.weight.data, 1, 0)
        # move the image to colmun shape for the weight and bais
        self.after_col_weight = new_weigth.reshape(self.output_channel, -1).T
        #print('w',self.col_w,self.col_w.shape)
        #print('x',self.col_x,self.col_x.shape)
        #col bias
        self.after_col_bais = self.bias.data.reshape(-1, self.output_channel)
        # matrix dot for the z = wx+b
        matirx_dot = np.dot(self.after_col_data, self.after_col_weight) + self.after_col_bais
        # reshape it back to origanl size
        reshape_matirx = matirx_dot.reshape(self.batch_size, self.output_height, self.output_width, -1)
        # reshape the axis for the data
        self.forward_data = np.transpose(reshape_matirx, axes=(0, 3, 1, 2))
        return self.forward_data

    #method
    # @staticmethod
    # @njit(parallel=True, cache=True)
    # def backward_numba(prepare_data, in_data, local_kh, local_kw, local_oh, local_ow, local_H, local_W, stride, padding):
    #     for row in range(local_kh):
    #         row_end = row + stride * local_oh
    #         for col in range(local_kw):
    #             col_end = col + stride *  local_ow
    #             prepare_data[:, :, row:row_end:stride, col:col_end:stride] += in_data[:, :, row, col, :, :] 
    #     back = prepare_data[:, :, padding:local_H + padding, padding:local_W + padding] 
    #     return back 
    
    def backward_numba(self, prepare_data, in_data, local_kh, local_kw, local_oh, local_ow, local_H, local_W, stride, padding):
        # for loop in row
        for row in range(local_kh):
            row_end = row + stride * local_oh
            # for loop in col
            for col in range(local_kw):
                col_end = col + stride *  local_ow
                # back col
                prepare_data[:, :, row:row_end:stride, col:col_end:stride] += in_data[:, :, row, col, :, :] 
        back_data = prepare_data[:, :, padding:local_H + padding, padding:local_W + padding] 
        return back_data


    def backward_helper(self, input_col, input_shape, local_kh, local_kw, local_oh, local_ow, stride, padding):
        #remember the shape
        local_N, local_C, local_H, local_W = input_shape
        # init a all 0 col for update
        init_col = input_col.reshape(local_N, local_oh, local_ow, local_C, local_kh, local_kw)
        #col to image
        col2data = np.transpose(init_col, axes=(0, 3, 4, 5, 1, 2))
        # inverse of padding 
        local_A = local_H + 2*padding + stride - 1
        local_B = local_W + 2*padding + stride - 1
        prepare_data = np.zeros((local_N, local_C, local_A, local_B))
        # cal the back
        backword_data = self.backward_numba(
            prepare_data, col2data, local_kh, 
            local_kw, local_oh, local_ow,local_H, 
            local_W, stride, padding
            )
        return backword_data

    def backward(self, previous_partial_gradient):
        deltain2col = np.transpose(previous_partial_gradient, axes=(0,2,3,1)).reshape(-1, self.output_channel)
        #print('grad',self.bias.grad,self.bias.grad.shape)
        # print('col——delta',col_delta_in,col_delta_in.shape)
        #inter = np.sum(col_delta_in, axis=0 ).T / self.batch_size
        #print('upgrad',inter,inter.shape)
        #self.bias.grad = np.sum(col_delta_in, axis=0, keepdims=True).T / self.batch_size

        #calculate the weight grad
        col_grad_W = np.dot(self.after_col_data.T, deltain2col)
        inter_gradw = np.transpose(col_grad_W, axes=(1, 0)).reshape(
            self.output_channel, self.input_channel, 
            self.kernel_height, self.kernel_width
            )
        self.weight.grad = np.moveaxis(inter_gradw, 1, 0)
        #calculate the bias grad
        self.bias.grad = np.sum(deltain2col, axis=0).T
        col_delta_out = np.dot(deltain2col, self.after_col_weight.T)
        # reshape it 
        delta_out = self.backward_helper(
            col_delta_out, self.input_data.shape, 
            self.kernel_height, self.kernel_width, 
            self.output_height, self.output_width, 
            self.stride, self.padding
            )
        return delta_out
    

    def selfstr(self):
        return "Kernel: (%s, %s) In Channels %s Out Channels %s Stride %s" % (
            self.weight.data.shape[2],
            self.weight.data.shape[3],
            self.weight.data.shape[0],
            self.weight.data.shape[1],
            self.stride,
        )

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(ConvLayer, self).initialize()