"""
    Utility modules of convolution operations.

    Author : @MGokcayK 
    Create : 30 / 04 / 2020
    Update : 27 / 08 / 2020
                Add get_conv3D_indices and update im2col_indices for new padding approach.
"""

import numpy as np


def get_im2col_indices(X_shape, kernel, stride, output):
    """    
        Implementation applied from : https://cs231n.github.io/convolutional-networks/#overview 
    of assingment 2.
    """
    # take dimension of X
    N, C, H, W = X_shape
    # get kernel shape of each axis
    HH = kernel[0]
    WW = kernel[1]
    # get stride of each axis
    stride_row = stride[0]
    stride_col = stride[1]
    # calculate output shape
    H_out, W_out = output[1], output[2]
    # calculate kernel locations
    i0 = np.repeat(np.arange(HH), WW)
    i0 = np.tile(i0, C)    
    i1 = stride_row * np.repeat(np.arange(H_out), W_out)    
    j0 = np.tile(np.arange(WW), HH * C)    
    j1 = stride_col * np.tile(np.arange(W_out), H_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), HH * WW).reshape(-1, 1)
    #print(j)

    return (k.astype(int), i.astype(int), j.astype(int))

def get_conv1D_indices(kernel, stride, output):
    """
        Written by @Author based on get_im2col_indices.
    """
    i0 = np.arange(kernel)
    i1 = stride * np.arange(output)   
    i = i1.reshape(-1, 1) + i0.reshape(1, -1)
    return i.astype(int)

def get_conv3D_indices(X_shape, kernel, stride, output):
    """
        Written by @Author based on get_im2col_indices.
    """
    # take some parameters from shape
    N, C, D, H, W = X_shape
    # get each parameter separately
    K_H, K_W, K_D = kernel[0], kernel[1], kernel[2]
    S_H, S_W, S_D = stride[0], stride[1], stride[2]
    # calculate output dims 
    O_D, O_H, O_W = output[1], output[2], output[3]
    # find locations 
    # height
    i0 = np.repeat(np.arange(K_H), K_W )
    i0 = np.tile(i0, C * K_D)
    i1 = S_H * np.tile(np.repeat(np.arange(O_H), O_W), O_D)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    # width
    j0 = np.tile(np.arange(K_W), K_H * K_D)
    j0 = np.tile(j0, C)
    j1 = S_W * np.tile(np.arange(O_W), O_H * O_D)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    # depth
    k0 = np.repeat(np.arange(K_D), K_H * K_W)
    k0 = np.tile(k0, C)
    k1 = S_D * np.repeat(np.arange(O_D), O_H * O_W)
    k = k0.reshape(-1, 1) + k1.reshape(1, -1)
    # channel 
    l = np.repeat(np.arange(C), K_H * K_W * K_D).reshape(-1, 1)

    return (l.astype(int), k.astype(int), i.astype(int), j.astype(int)) 