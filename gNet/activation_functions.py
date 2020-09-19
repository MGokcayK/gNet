"""
    Activation function module of gNet.

    Containing activation function methods (with calling string) : \n
        - Rectified Linear Unit Function, Relu ('relu')
        - Leaky Rectified Linear Unit Function, LRelu ('lrelu')
        - Sigmoid, ('sigmoid')
        - Softmax, ('softmax')
        - Softplus ('softplus')
        - Tanh, ('tanh')

    To select proper activation function for model user can describe in layer addition. \n
        Ex :
        ```python
            ...
            net = NeuralNetwork()
            net.add_Dense(32, activation_function = 'softmax')
            ...
        ```

    Author : @MGokcayK 
    Create : 28 / 03 / 2020
    Update : 19 / 09 / 2020
                Alter tanh activation function.
"""

import numpy as np
import gNet.tensor as T

class ActivationFunction:
    """
        Base class of activation function.

        To implement new activation function, developer should implement activate
        function which calculate the function.

        Containing activation function methods (with calling string) : \n
            - Rectified Linear Unit Function, Relu ('relu')
            - Leaky Rectified Linear Unit Function, LRelu ('lrelu')
            - Sigmoid, ('sigmoid')
            - Softmax, ('softmax')
            - Softplus ('softplus')
            - Tanh, ('tanh')
        
        To select proper activation function for model user can describe in layer addition. \n
            Ex :
                ...
                >>> net = NeuralNetwork()
                    net.add_ Dense(32, activation_function = 'softmax')
                ...
    """
    @staticmethod
    def activate(x):
        raise NotImplementedError


class Relu(ActivationFunction):
    """
        Implementation of Rectified Linear Unit Function.

        Arguments :
        -----------

        x   : Value to activate w.r.t function.

        >>> Relu = max(x, 0)
    """  
    @staticmethod
    def activate(x):  
        x = T.where(x, x.value > 0, x, 0)
        return x

class LeakyRelu(ActivationFunction):
    """
        Implementation of Leaky Rectified Linear Unit Function.

        Arguments :
        -----------

        x   : Value to activate w.r.t function.

        >>> Relu = max(x, 0.01x)
    """  
    @staticmethod
    def activate(x):  
        x = T.where(x, x.value > 0, x, 0.01*x)
        return x

class Softplus(ActivationFunction):
    """
        Implementation of Softplus.

        Arguments :
        -----------

        x   : Value to activate w.r.t function.

        >>> Softplut = log(1.0 + exp(x))
    """  
    @staticmethod
    def activate(x):  
        x = T.log(1. + T.exp(x))
        return x

class Sigmoid(ActivationFunction):
    """
        Implementation of Sigmoid Function.

        Arguments :
        -----------

        x   : Value to activate w.r.t function.

        >>> Sigmoid = 1.0 / (1.0 + exp(-x))
    """
    @staticmethod
    def activate(x):
        return 1.0 / (1.0 + T.exp(-x))

class Softmax(ActivationFunction):
    """
        Implementation of Softmax Function.

        Arguments :
        -----------

        x   : Value to activate w.r.t function.

        >>> Softmax = exp(x) / SUM(exp(x))
    """
    @staticmethod
    def activate(x):
        ''' Implementation of Stable Softmax Function.'''  
        # Finding max of tensor to have mathematical stability.
        m = T.make_tensor(np.max(x.value, axis=-1, keepdims=True))
        # Substitute of the max values.
        a = x - m 
        # Taking exponent of `Tensor`.
        b = T.exp(a)
        # Normalize
        c = b / T.tensor_sum(b, axis=-1, keepdim=True)
        return c

class Tanh(ActivationFunction):
    """
        Activation of Tanh Function.

        Arguments :
        -----------

        x   : Value to activate w.r.t function.

        >>> Tanh = (exp(2x) - 1) / (exp(2x) + 1) = 1.0 - 2.0 / (exp(2*x) + 1.)
    """
    @staticmethod
    def activate(x):
        return T.tanh(x)

class noActivate(ActivationFunction):
    """
        No activate function. If layer is not declared, it will be noActive function.

        Arguments :
        -----------

        x   : Value to activate w.r.t function.

        >>> noActivae = x
    """
    @staticmethod
    def activate(x):
        return x


__activationFunctionsDecleration = {
                                    'relu' : Relu, 
                                    'lrelu': LeakyRelu,
                                    'sigmoid': Sigmoid,
                                    'softmax': Softmax,
                                    'tanh' : Tanh,
                                    'softplus' : Softmax,
                                    'none' : noActivate
                                 }