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
    Update : 16 / 03 / 2021
                Added custom AFDecleration Dictionary, 
                KeyError handling mechanism,
                Registering Custom Activation Function property
                and check type of registering custom class type.
"""

import numpy as np
import gNet.tensor as T

class _AFDeclarationDict(dict):
    """
        Custom Dictionary class for Activation Functions. It stores registered 
        activation functions.
    """
    def __getitem__(self, key):
        try:
            return self.__dict__[key]
        except KeyError as e:
            msg = f"The Activation Function `{key}` is not registered! " 
            msg += "Please use `REGISTER_ACTIVATION_FUNCTION` method for registering."
            raise KeyError(msg) from None
  
    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)


__activationFunctionsDecleration = _AFDeclarationDict()


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
        x = T.maximum(x, 0)
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
        x = T.maximum(x, 0.01 * x)
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



def REGISTER_ACTIVATION_FUNCTION(act_function : ActivationFunction, call_name : str):
    """
        Register Activation Function w.r.t `call_name`. 

        Arguments :
        -----------

        act_function    : Activation function class.
        >>>    type     : gNet.activation_functions.ActivationFunction()

        call_name       : Calling name of act. function. It will be lowercase. It is not sensible.
        >>>    type     : str
    """
    if isinstance(act_function(), ActivationFunction):
        __activationFunctionsDecleration.update({call_name.lower() : act_function})
    else:
        msg = f"The Activation Function Class `{act_function}` is not same as gNet's Activation Function base class. " 
        msg += f"\nPlease make sure that `{act_function}` inherit from gNet.activation_functions.ActivationFunction"
        raise TypeError(msg)


REGISTER_ACTIVATION_FUNCTION(Relu, 'relu')
REGISTER_ACTIVATION_FUNCTION(LeakyRelu, 'lrelu')
REGISTER_ACTIVATION_FUNCTION(Sigmoid, 'sigmoid')
REGISTER_ACTIVATION_FUNCTION(Softmax, 'softmax')
REGISTER_ACTIVATION_FUNCTION(Tanh, 'tanh')
REGISTER_ACTIVATION_FUNCTION(Softplus, 'softplus')
REGISTER_ACTIVATION_FUNCTION(noActivate, 'none')
