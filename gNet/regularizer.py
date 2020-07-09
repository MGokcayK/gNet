"""
    Regularizer module of gNet.

    Containing regularizer methods: \n
        - L1
        - L2
        - L1L2 (Containing two of them at the same time)

    To use regularizer for training: \n
        - User can use regularizer in layer directly.
            
            Ex: 
            from gNet.regularizer import L2
            >>> model = gNet.Model.Model()
            ...
            >>> model.add(lyr.Dense(128,'relu', kernel_regularizer=L2()))
            ...

    Author : @MGokcayK 
    Create : 26 / 06 / 2020
    Update : 26 / 06 / 2020
                Creating module and methods.
"""



import numpy as np
import gNet.tensor as T

class Regularizer:
    """
        Base Regularizer class of gNet.

        Regularization is penalty of system. gNet can calculate kernel_regularizer and 
        bias_regularizer in layers which has regularization parameters like Dense, Conv2D. 

        Containing regularizer methods: \n
            - L1
            - L2
            - L1L2 (Containing two of them at the same time)

        To implement new regularizer, developer need to add compute method to 
        compute regularization parameter.

        The important point of compute method is parameter should have only one
        argument. 

        Ex. L2 compute method: 
        
        >>> def compute(self, parameter: 'Tensor') -> 'Tensor':
                self._regularizer = self._lmb * T.power(T.make_tensor(parameter), 2).sum()
                return self._regularizer

        To use regularizer for training: \n
            - User can use regularizer in layer directly.
                
                Ex: 
                from gNet.regularizer import L2
                >>> model = gNet.Model.Model()
                ...
                >>> model.add(lyr.Dense(128,'relu', kernel_regularizer=L2()))
                ...
    """


    def compute(self, parameter: 'Tensor') -> 'Tensor':
        raise NotImplementedError

class L1(Regularizer):
    """
        L1 Regularization method. It computes formula of:
            r = \lambda * sum (|x|) 
                where x is parameter.

        Arguments:
        ----------

        Lmb     : Lambda value of penalty.
            >>> type    : float
            >>> Default : 0.01
    """

    def __init__(self, Lmb = 0.01, **kwargs):
        super(L1, self).__init__(**kwargs)
        self._lmb = Lmb

    def compute(self, parameter: 'Tensor') -> 'Tensor':
        self._regularizer = self._lmb * T.abs(T.make_tensor(parameter)).sum()
        return self._regularizer
        
class L2(Regularizer):
    """
        L2 Regularization method. It computes formula of:
            r = \lambda * sum (x**2)
                where x is parameter.

        Arguments:
        ----------

        Lmb     : Lambda value of penalty.
            >>> type    : float
            >>> Default : 0.01
    """

    def __init__(self, Lmb = 0.01, **kwargs):
        super(L2, self).__init__(**kwargs)
        self._lmb = Lmb

    def compute(self, parameter: 'Tensor') -> 'Tensor':
        self._regularizer = self._lmb * T.power(T.make_tensor(parameter), 2).sum()
        return self._regularizer

class L1L2(Regularizer):
    """
        L1L2 is Regularization method of two method. It computes formula of:
            r = \L1 * sum (|x|) + \L2 * sum(x**2)

        Arguments:
        ----------

        L1     : Lambda value of penalty L1.
            >>> type    : float
            >>> Default : 0.01

        L2     : Lambda value of penalty L2.
            >>> type    : float
            >>> Default : 0.01
    """

    def __init__(self, L1 = 0.01, L2 = 0.01, **kwargs):
        super(L1L2, self).__init__(**kwargs)
        self._L1 = L1
        self._L2 = L2

    def compute(self, parameter: 'Tensor') -> 'Tensor':
        self._regularizer = self._L1 * T.abs(T.make_tensor(parameter)).sum()
        self._regularizer += self._L2 * T.power(T.make_tensor(parameter), 2).sum()
        return self._regularizer
