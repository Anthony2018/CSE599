def backward(self, previous_partial_gradient):
        #(output_h, output_w)= self.calculate_output_size(self.input_height,self.input_width,self.filter_height,self.filter_width,self.padding,self.stride)
        # expand the maxtrix if the stride is not 1
        expand_h = 0
        expand_w = 0
        if self.stride == 1:
            dZ_stride_1 = previous_partial_gradient
            expand_h = dZ_stride_1[2]
            expand_w = dZ_stride_1[3]
        else:
            (expand_h,expand_w) = self.calculate_output_size(self.input_height,self.input_width,self.filter_height,self.filter_width,self.padding,1)
            dZ_stride_1 = np.zeros((self.batch_size, self.input_channel, expand_h, expand_w))
            for bs in prange(self.batch_size):
                for ic in prange(self.input_channel):
                    for i in prange(output_h):
                        for j in prange(output_w):
                            ii = i * self.stride
                            jj = j * self.stride
                            dZ_stride_1[bs, ic, ii, jj] = data[bs, ic, i, j]
        #update the weight and bias
        pad_h = ((output_h - 1) * self.stride - input_height + filter_height) // 2
        pad_w = ((output_w - 1) * self.stride - input_width + filter_width) // 2
        input_padded = np.pad(self.data, ((0,0),(0,0),(pad_h, pad_h),(pad_w,pad_w)), 'constant')
        for bs in prange(batch_size):
            for oc in prange(output_channel):   # == kernal count
                for ic in prange(input_channel):    # == filter count
                    w_grad = np.zeros((filter_height, filter_width))
                    self.jit_conv_2d(input_padded[bs,ic], dZ_stride_1[bs,oc], 0, kernel_grad)
                    self.Weight.grad[oc,ic] += w_grad
                self.bias.grad[oc] += dZ_stride_1[bs,oc].sum()
        #mean the grad
        self.weight.grad = self.weight.grad/batch_size
        self.bias.grad = self.bias.grad/batch_size
        #update the grad
        dz_padded = np.pad(dZ_stride_1, ((0,0),(0,0),(pad_h, pad_h),(pad_w, pad_w)), 'constant')
        rot_weights = self.weight.data.T
        delta_out = np.zeros(self.data.shape)
        for bs in prange(batch_size):
            for oc in prange(output_channel):    # == kernal count
                delta_per_input = np.zeros((input_height, input_width)).astype(np.float64)
                #delta_per_input = np.zeros((input_height, input_width))
                for ic in range(input_channel): # == filter count
                    self.jit_conv_2d(dz_padded[bs,oc], rot_weights[oc,ic], 0, delta_per_input)
                    delta_out[bs,ic] += delta_per_input
        return delta_out
        delta_out = self.backward_numba(previous_partial_gradient,self.data,self.weight, self.weight.grad)
        return delta_out