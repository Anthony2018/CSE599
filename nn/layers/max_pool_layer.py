import numbers

import numpy as np
from numba import njit, prange

from .layer import Layer


class MaxPoolLayer(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2, parent=None):
        super(MaxPoolLayer, self).__init__(parent)
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
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
    def max_forward_numba(self, prepare_data, local_kh, local_kw,local_oh,local_ow, stride, pre_col):
        for row in range(local_kh):
            row_end = row + stride*local_oh
            for col in range(local_kw):
                col_end = col + stride*local_ow
                pre_col[:, :, row, col, :, :] = prepare_data[:, :, row:row_end:stride, col:col_end:stride]
        return pre_col

            
    def max_forward_helper(self, input_data, kernel_h, kernel_w, stride=1, padding=0):
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
        mid_col = self.max_forward_numba(
            prepare_data, kernel_h, 
            kernel_w,local_out_h,
            local_out_w, stride, pre_col
            )
        # reshpe back
        reshape_size = local_N*local_out_h*local_out_w
        final_col = np.transpose(mid_col, axes=(0, 4, 5, 1, 2, 3)).reshape(
            reshape_size, -1
            )
        return final_col

    def forward(self, data):
        # remember the data and different size
        # remember the input_data
        self.input_data = data
        # remember the batch size
        self.batch_size = self.input_data.shape[0]
        # remember the channel
        self.input_channel = self.input_data.shape[1]
        # input size
        self.input_height= self.input_data.shape[2]
        self.input_width = self.input_data.shape[3]
        # output size
        self.kernel_height = self.kernel_size
        self.kernel_width= self.kernel_size
        # calclulate the output size by padding.
        self.output_height = (self.input_height - self.kernel_height + 2 * self.padding) // self.stride + 1    
        self.output_width = (self.input_width - self.kernel_width + 2 * self.padding) // self.stride + 1
        after_col_data = self.max_forward_helper(
            self.input_data, self.kernel_height, 
            self.kernel_width, self.stride, 
            self.padding
            )
        reshape_col_x = after_col_data.reshape(-1, self.kernel_width * self.kernel_height)
        self.col_max_data = np.argmax(reshape_col_x, axis=1)
        repshape_max = np.max(reshape_col_x, axis=1)
        data_shape_max = repshape_max.reshape(
            self.batch_size, self.output_height, 
            self.output_width, self.input_channel)
        self.forward_data = np.transpose(data_shape_max, axes=(0,3,1,2))
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
            row_end = stride * local_oh + row  
            # for loop in col
            for col in range(local_kw):
                col_end = stride *  local_ow + col
                # back col
                prepare_data[:, :, row:row_end:stride, col:col_end:stride] += in_data[:, :, row, col, :, :] 
        back = prepare_data[:, :, padding:local_H + padding, padding:local_W + padding] 
        return back 


    def backward_helper(self, input_col, input_shape, local_kh, local_kw, local_oh, local_ow, stride, padding):
        #remember the shape
        local_N, local_C, local_H, local_W = input_shape
        # init a all 0 col for update
        init_col = input_col.reshape(
            local_N, local_oh, local_ow, 
            local_C, local_kh, local_kw
            )
        #col to image
        col2data = np.transpose(init_col, axes=(0, 3, 4, 5, 1, 2))
        prepare_data = np.zeros((
            local_N, local_C, 
            local_H + 2*padding + stride - 1, 
            local_W + 2*padding + stride - 1
            ))
        # cal the back
        backword_data = self.backward_numba(
            prepare_data, col2data, local_kh, 
            local_kw, local_oh, local_ow,local_H, 
            local_W, stride, padding
            )
        return backword_data


    def backward(self, previous_partial_gradient):
        # cal the pool size of all
        self.col_pool = self.kernel_width * self.kernel_height
        deltain2col = np.transpose(previous_partial_gradient, (0,2,3,1))
        #inital for update
        delta_col_max = np.zeros((deltain2col.size, self.col_pool))
        flatten = deltain2col.flatten()
        delta_col_max[np.arange(self.col_max_data.size), self.col_max_data.flatten()] = flatten
        delta_col_max = delta_col_max.reshape(deltain2col.shape + (self.col_pool,))
        A = delta_col_max.shape[0]
        B = delta_col_max.shape[1]
        C = delta_col_max.shape[2]
        reshape_size = A * B * C
        delta_col_final = delta_col_max.reshape(reshape_size, -1)
        back_grad = self.backward_helper(
            delta_col_final, self.input_data.shape, 
            self.kernel_height, self.kernel_width, 
            self.output_height, self.output_width, 
            self.stride, self.padding
            )
        return back_grad

    def selfstr(self):
        return str("kernel: " + str(self.kernel_size) + " stride: " + str(self.stride))