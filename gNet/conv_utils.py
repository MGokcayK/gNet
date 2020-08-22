"""
    Utility modules of convolution operations.

    Implementation applied from : https://cs231n.github.io/convolutional-networks/#overview 
    of assingment 2.

    Author : @MGokcayK 
    Create : 30 / 04 / 2020
    Update : 03 / 07 / 2020
                Add descriptions.
"""

import numpy as np


def get_im2col_indices(X_shape, kernel, stride, padding):
    # take dimension of X
    N, C, H, W = X_shape
    # get kernel shape of each axis
    HH = kernel[0]
    WW = kernel[1]
    # get stride of each axis
    stride_row = stride[0]
    stride_col = stride[1]
    # calculate output shape
    H_out = int((H - HH + 2 * padding) / stride_row + 1)
    W_out = int((W - WW + 2 * padding) / stride_col + 1)
    # calculate kernel locations
    i0 = np.repeat(np.arange(HH), WW)
    i0 = np.tile(i0, C)    
    i1 = stride_row * np.repeat(np.arange(H_out), W_out)    
    j0 = np.tile(np.arange(HH), WW * C)    
    j1 = stride_col * np.tile(np.arange(W_out), H_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), HH * WW).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))

def get_conv1D_indices(kernel, stride, output):
    i0 = np.arange(kernel)
    i1 = stride * np.arange(output)   
    i = i1.reshape(-1, 1) + i0.reshape(1, -1)
    return i
