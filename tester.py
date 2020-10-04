"""
    Tester script of gNet.

    Author : @MGokcayK 
    Create : 04 / 09 / 2020
    Update : 04 / 09 / 2020
                Testing cpp_add_ops and its grad calculations.
"""

import gNet.test.py_cpp_ops_test as tester
import gNet.test.py_wrapper_test as wrapper

#wrapper.wrapper_test()

#tester.test_add_ops()
#tester.test_add_ops_grad1()
tester.test_add_ops_grad2()