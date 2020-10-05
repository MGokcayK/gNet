"""
    Tester script of gNet_CPP.

    Author : @MGokcayK 
    Create : 04 / 09 / 2020
    Update : 04 / 09 / 2020
                Testing cpp_ops and its grad calculations.
"""

import ctypes
from ctypes import c_bool, c_float
import numpy as np
import numpy.ctypeslib as ctl
import gNet.tensor as T
import os 

so_location= os.path.dirname(os.path.dirname(__file__)) + "\\bin"
so_name = "libcpp_ops_test.so"
soabspath = so_location + os.path.sep + so_name


def test_add_ops():    
    
    cpp_so = ctypes.CDLL(soabspath) # calling shared library

    d = np.array(np.random.randn(1,3), dtype=np.float32) # create random array
    d2 = np.array(np.random.randn(3,3), dtype=np.float32) 

    # make gNet Tensor
    t1 = T.Tensor(d, True)
    t2 = T.Tensor(d2, True)

    # do ops
    r = T.add(t1, t2) 
    r.backward()
    
    # calling cpp functions and set its arguments and result types
    cpp_ops = cpp_so.cpp_ops_add_test
    cpp_ops.argtypes = [ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
                        ctypes.c_bool,
                        ctypes.c_int,
                        ctypes.c_int * len(d.shape), 
                        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
                        ctypes.c_bool,
                        ctypes.c_int,
                        ctypes.c_int * len(d2.shape),
                        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'),
                        ctypes.c_int,
                        ctypes.c_int * len(r.value.shape)]
    
    cpp_ops(d, t1.have_grad, d.ndim, d.ctypes.shape_as(ctypes.c_long), d2, t2.have_grad, d2.ndim, d2.ctypes.shape_as(ctypes.c_long), 
                r.value, r.value.ndim, r.value.ctypes.shape_as(ctypes.c_long))



def test_add_ops_grad1():    
    
    cpp_so = ctypes.CDLL(soabspath) # calling shared library

    d = np.array(np.random.randn(3,1), dtype=np.float32) # create random array
    d2 = np.array(np.random.randn(3,3), dtype=np.float32) 

    # make gNet Tensor
    t1 = T.Tensor(d, True)
    t2 = T.Tensor(d2, True)

    # do ops
    r = T.add(t1, t2) 
    r.backward()
    
    # calling cpp functions and set its arguments and result types
    cpp_ops = cpp_so.cpp_ops_add_grad_test1
    cpp_ops.argtypes = [ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
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
    
    cpp_ops(d, t1.have_grad, d.ndim, d.ctypes.shape_as(ctypes.c_long), d2, t2.have_grad, d2.ndim, d2.ctypes.shape_as(ctypes.c_long), 
                t1.grad.value.astype(np.float32), t1.grad.value.ndim, t1.grad.value.ctypes.shape_as(ctypes.c_long))

def test_add_ops_grad2():    
    
    cpp_so = ctypes.CDLL(soabspath) # calling shared library

    d = np.array(np.random.randn(3,1), dtype=np.float32) # create random array
    d2 = np.array(np.random.randn(3,3), dtype=np.float32) 

    # make gNet Tensor
    t1 = T.Tensor(d, True)
    t2 = T.Tensor(d2, True)

    # do ops
    r = T.add(t1, t2) 
    r.backward()
    
    # calling cpp functions and set its arguments and result types
    cpp_ops = cpp_so.cpp_ops_add_grad_test2
    cpp_ops.argtypes = [ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
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
    
    cpp_ops(d, t1.have_grad, d.ndim, d.ctypes.shape_as(ctypes.c_long), d2, t2.have_grad, d2.ndim, d2.ctypes.shape_as(ctypes.c_long), 
                t2.grad.value.astype(np.float32), t2.grad.value.ndim, t2.grad.value.ctypes.shape_as(ctypes.c_long))


def test_matmul_ops():    
    
    cpp_so = ctypes.CDLL(soabspath) # calling shared library

    d = np.array(np.random.randn(2,3), dtype=np.float32) # create random array
    d2 = np.array(np.random.randn(3,4), dtype=np.float32) 

    # make gNet Tensor
    t1 = T.Tensor(d, True)
    t2 = T.Tensor(d2, True)

    # do ops
    r = T.matmul(t1, t2) 
    r.backward()
    
    # calling cpp functions and set its arguments and result types
    cpp_ops = cpp_so.cpp_ops_matmul_test
    cpp_ops.argtypes = [ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
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
    
    cpp_ops(d, t1.have_grad, d.ndim, d.ctypes.shape_as(ctypes.c_long), d2, t2.have_grad, d2.ndim, d2.ctypes.shape_as(ctypes.c_long), 
                r.value, r.value.ndim, r.value.ctypes.shape_as(ctypes.c_long))


def test_matmul_ops_grad1():    
    
    cpp_so = ctypes.CDLL(soabspath) # calling shared library

    d = np.array(np.random.randn(2,3), dtype=np.float32) # create random array
    d2 = np.array(np.random.randn(3,4), dtype=np.float32) 

    # make gNet Tensor
    t1 = T.Tensor(d, True)
    t2 = T.Tensor(d2, True)

    # do ops
    r = T.matmul(t1, t2) 
    r.backward()
    
    # calling cpp functions and set its arguments and result types
    cpp_ops = cpp_so.cpp_ops_matmul_grad_test1
    cpp_ops.argtypes = [ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
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
    
    cpp_ops(d, t1.have_grad, d.ndim, d.ctypes.shape_as(ctypes.c_long), d2, t2.have_grad, d2.ndim, d2.ctypes.shape_as(ctypes.c_long), 
                t1.grad.value.astype(np.float32), t1.grad.value.ndim, t1.grad.value.ctypes.shape_as(ctypes.c_long))

def test_matmul_ops_grad2():    
    
    cpp_so = ctypes.CDLL(soabspath) # calling shared library

    d = np.array(np.random.randn(2,3), dtype=np.float32) # create random array
    d2 = np.array(np.random.randn(3,4), dtype=np.float32) 

    # make gNet Tensor
    t1 = T.Tensor(d, True)
    t2 = T.Tensor(d2, True)

    # do ops
    r = T.matmul(t1, t2) 
    r.backward()
    
    # calling cpp functions and set its arguments and result types
    cpp_ops = cpp_so.cpp_ops_matmul_grad_test2
    cpp_ops.argtypes = [ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
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
    
    cpp_ops(d, t1.have_grad, d.ndim, d.ctypes.shape_as(ctypes.c_long), d2, t2.have_grad, d2.ndim, d2.ctypes.shape_as(ctypes.c_long), 
                t2.grad.value.astype(np.float32), t2.grad.value.ndim, t2.grad.value.ctypes.shape_as(ctypes.c_long))


def test_mul_ops():    
    
    cpp_so = ctypes.CDLL(soabspath) # calling shared library

    d = np.array(np.random.randn(5,3,3), dtype=np.float32) # create random array
    d2 = np.array(np.random.randn(1,3,3), dtype=np.float32) 

    # make gNet Tensor
    t1 = T.Tensor(d, True)
    t2 = T.Tensor(d2, True)

    # do ops
    r = T.mul(t1, t2) 
    r.backward()

    # calling cpp functions and set its arguments and result types
    cpp_ops = cpp_so.cpp_ops_mul_test
    cpp_ops.argtypes = [ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
                        ctypes.c_bool,
                        ctypes.c_int,
                        ctypes.c_int * len(d.shape), 
                        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
                        ctypes.c_bool,
                        ctypes.c_int,
                        ctypes.c_int * len(d2.shape),
                        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'),
                        ctypes.c_int,
                        ctypes.c_int * len(r.shape)]
    
    cpp_ops(d, t1.have_grad, d.ndim, d.ctypes.shape_as(ctypes.c_long), d2, t2.have_grad, d2.ndim, d2.ctypes.shape_as(ctypes.c_long), 
                r.value, r.value.ndim, r.value.ctypes.shape_as(ctypes.c_long))


def test_mul_ops_grad1():    
    
    cpp_so = ctypes.CDLL(soabspath) # calling shared library

    d = np.array(np.random.randn(2,3), dtype=np.float32) # create random array
    d2 = np.array(np.random.randn(1,3), dtype=np.float32) 

    # make gNet Tensor
    t1 = T.Tensor(d, True)
    t2 = T.Tensor(d2, True)

    # do ops
    r = T.mul(t1, t2) 
    r.backward()
    
    # calling cpp functions and set its arguments and result types
    cpp_ops = cpp_so.cpp_ops_mul_grad_test1
    cpp_ops.argtypes = [ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
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
    
    cpp_ops(d, t1.have_grad, d.ndim, d.ctypes.shape_as(ctypes.c_long), d2, t2.have_grad, d2.ndim, d2.ctypes.shape_as(ctypes.c_long), 
                t1.grad.value.astype(np.float32), t1.grad.value.ndim, t1.grad.value.ctypes.shape_as(ctypes.c_long))

def test_mul_ops_grad2():    
    
    cpp_so = ctypes.CDLL(soabspath) # calling shared library

    d = np.array(np.random.randn(3), dtype=np.float32) # create random array
    d2 = np.array(np.random.randn(2,3), dtype=np.float32) 

    # make gNet Tensor
    t1 = T.Tensor(d, True)
    t2 = T.Tensor(d2, True)

    # do ops
    r = T.mul(t1, t2) 
    r.backward()
    
    # calling cpp functions and set its arguments and result types
    cpp_ops = cpp_so.cpp_ops_mul_grad_test2
    cpp_ops.argtypes = [ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
                        ctypes.c_bool,
                        ctypes.c_int,
                        ctypes.c_int * len(d.shape), 
                        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
                        ctypes.c_bool,
                        ctypes.c_int,
                        ctypes.c_int * len(d2.shape),
                        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'),
                        ctypes.c_int,
                        ctypes.c_int * len(t2.grad.shape)]
    
    cpp_ops(d, t1.have_grad, d.ndim, d.ctypes.shape_as(ctypes.c_long), d2, t2.have_grad, d2.ndim, d2.ctypes.shape_as(ctypes.c_long), 
                t2.grad.value.astype(np.float32), t2.grad.value.ndim, t2.grad.value.ctypes.shape_as(ctypes.c_long))



def test_pow_ops():    
    
    cpp_so = ctypes.CDLL(soabspath) # calling shared library

    d = np.abs(np.array(np.random.randn(2,3), dtype=np.float32)) # create random array
    d2 = float(2/3)

    # make gNet Tensor
    t1 = T.Tensor(d, True)
    
    # do ops
    r = T.power(t1, d2)
    r.backward()

    # calling cpp functions and set its arguments and result types
    cpp_ops = cpp_so.cpp_ops_pow_test
    cpp_ops.argtypes = [ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
                        ctypes.c_bool,
                        ctypes.c_int,
                        ctypes.c_int * len(d.shape), 
                        ctypes.c_float, 
                        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'),
                        ctypes.c_int,
                        ctypes.c_int * len(r.shape)]
    
    cpp_ops(d, t1.have_grad, d.ndim, d.ctypes.shape_as(ctypes.c_long), d2,
                r.value, r.value.ndim, r.value.ctypes.shape_as(ctypes.c_long))


def test_pow_ops_grad1():    
    
    cpp_so = ctypes.CDLL(soabspath) # calling shared library

    d = np.abs(np.array(np.random.randn(2,3), dtype=np.float32)) # create random array
    d2 = float(4)

    # make gNet Tensor
    t1 = T.Tensor(d, True)

    # do ops
    r = T.power(t1, d2) 
    r.backward()
    
    # calling cpp functions and set its arguments and result types
    cpp_ops = cpp_so.cpp_ops_mul_grad_test1
    cpp_ops.argtypes = [ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
                        ctypes.c_bool,
                        ctypes.c_int,
                        ctypes.c_int * len(d.shape), 
                        ctypes.c_float,
                        ctl.ndpointer(np.float32, flags='aligned, c_contiguous'),
                        ctypes.c_int,
                        ctypes.c_int * len(t1.grad.value.shape)]
    
    cpp_ops(d, t1.have_grad, d.ndim, d.ctypes.shape_as(ctypes.c_long), d2, 
                t1.grad.value.astype(np.float32), t1.grad.value.ndim, t1.grad.value.ctypes.shape_as(ctypes.c_long))