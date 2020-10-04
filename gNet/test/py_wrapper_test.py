import ctypes
"""
    Test wrapper script of gNet_py.

    Author : @MGokcayK 
    Create : 04 / 09 / 2020
    Update : 04 / 09 / 2020
                Testing wrapper functions of gNet_py which shows wheter data flow correctly or not.
"""

import numpy as np
import numpy.ctypeslib as ctl
import os 


def wrapper_test():
    so_location= os.path.dirname(os.path.dirname(__file__)) + "\\bin"
    so_name = "libpy_wrapper.so"
    soabspath = so_location + os.path.sep + so_name
    
    py_wrapper_so = ctypes.CDLL(soabspath)


    d = np.array(np.random.randn(3,5,4), dtype=np.float32)

    py_wrapper_func = py_wrapper_so.py_wrapper_test
    py_wrapper_func.argtypes = [ctl.ndpointer(np.float32, flags='aligned, c_contiguous'), 
                        ctypes.c_int,
                        ctypes.c_int * len(d.shape)]

    py_wrapper_func(d, d.ndim, d.ctypes.shape_as(ctypes.c_long))
