"""
    Tester script of gNet_CPP.

    Author : @MGokcayK 
    Create : 04 / 09 / 2020
    Update : 04 / 09 / 2020
                Testing cpp_add_ops and its grad calculations.
"""

import ctypes
from ctypes import c_bool
import numpy as np
import numpy.ctypeslib as ctl
import gNet.tensor as T
import os 

so_location= os.path.dirname(os.path.dirname(__file__)) + "\\bin"
so_name = "libcpp_ops_test.so"
soabspath = so_location + os.path.sep + so_name


def test_add_ops():    
    
    cpp_ops = ctypes.CDLL(soabspath) # calling shared library

    d = np.array(np.random.randn(3,1), dtype=np.float32) # create random array
    d2 = np.array(np.random.randn(3,3), dtype=np.float32) 

    # make gNet Tensor
    t1 = T.Tensor(d, True)
    t2 = T.Tensor(d2, True)

    # do ops
    r = T.add(t1, t2) 
    r.backward()
    
    # calling cpp functions and set its arguments and result types
    cpp_add_ops = cpp_ops.cpp_ops_add_test
    cpp_add_ops.argtypes = [ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
                        ctypes.c_bool,
                        ctypes.c_int,
                        ctypes.c_int * len(d.shape), 
                        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
                        ctypes.c_bool,
                        ctypes.c_int,
                        ctypes.c_int * len(d2.shape),
                        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'),
                        ctypes.c_int,
                        ctypes.c_int * len(t1.grad.shape)]
    
    cpp_add_ops(d, t1.have_grad, d.ndim, d.ctypes.shape_as(ctypes.c_long), d2, t2.have_grad, d2.ndim, d2.ctypes.shape_as(ctypes.c_long), 
                r.value, r.value.ndim, r.value.ctypes.shape_as(ctypes.c_long))



def test_add_ops_grad1():    
    
    cpp_ops = ctypes.CDLL(soabspath) # calling shared library

    d = np.array(np.random.randn(3,1), dtype=np.float32) # create random array
    d2 = np.array(np.random.randn(3,3), dtype=np.float32) 

    # make gNet Tensor
    t1 = T.Tensor(d, True)
    t2 = T.Tensor(d2, True)

    # do ops
    r = T.add(t1, t2) 
    r.backward()
    
    # calling cpp functions and set its arguments and result types
    cpp_add_ops = cpp_ops.cpp_ops_add_grad_test1
    cpp_add_ops.argtypes = [ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
                        ctypes.c_bool,
                        ctypes.c_int,
                        ctypes.c_int * len(d.shape), 
                        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
                        ctypes.c_bool,
                        ctypes.c_int,
                        ctypes.c_int * len(d2.shape),
                        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'),
                        ctypes.c_int,
                        ctypes.c_int * len(t1.grad.shape)]
    
    cpp_add_ops(d, t1.have_grad, d.ndim, d.ctypes.shape_as(ctypes.c_long), d2, t2.have_grad, d2.ndim, d2.ctypes.shape_as(ctypes.c_long), 
                t1.grad.value.astype(np.float32), t1.grad.value.ndim, t1.grad.value.ctypes.shape_as(ctypes.c_long))

def test_add_ops_grad2():    
    
    cpp_ops = ctypes.CDLL(soabspath) # calling shared library

    d = np.array(np.random.randn(3,1), dtype=np.float32) # create random array
    d2 = np.array(np.random.randn(3,3), dtype=np.float32) 

    # make gNet Tensor
    t1 = T.Tensor(d, True)
    t2 = T.Tensor(d2, True)

    # do ops
    r = T.add(t1, t2) 
    r.backward()
    
    # calling cpp functions and set its arguments and result types
    cpp_add_ops = cpp_ops.cpp_ops_add_grad_test2
    cpp_add_ops.argtypes = [ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
                        ctypes.c_bool,
                        ctypes.c_int,
                        ctypes.c_int * len(d.shape), 
                        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
                        ctypes.c_bool,
                        ctypes.c_int,
                        ctypes.c_int * len(d2.shape),
                        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'),
                        ctypes.c_int,
                        ctypes.c_int * len(t1.grad.shape)]
    
    cpp_add_ops(d, t1.have_grad, d.ndim, d.ctypes.shape_as(ctypes.c_long), d2, t2.have_grad, d2.ndim, d2.ctypes.shape_as(ctypes.c_long), 
                t2.grad.value.astype(np.float32), t2.grad.value.ndim, t2.grad.value.ctypes.shape_as(ctypes.c_long))
