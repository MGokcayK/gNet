"""
    Optimizer module of gNet.

    Containing optimizer methods with calling strings : \n
        - SGD ('sgd')
        - Adagrad ('adagrad')
        - RMSprop ('rmsprop')
        - AdaDelta ('adadelta')
        - Adam ('adam')

    To call optimizer for training, user have two way. \n
        - User can define optimizer as string in calling NeuralNetwork.setup() function.
            
            Ex: 
                NeuralNetwork.setup(loss_function='categoricalcrossentropy', optimizer='SGD')

        - User can define optimizer explicitly.
            
            Ex:
                opt = gNet.optimizer.SGD(lr=0.001, momentum = 0.5)
                NeuralNetwork.setup(loss_function='categoricalcrossentropy', optimizer=opt)  

    Author : @MGokcayK 
    Create : 28 / 03 / 2020
    Update : 16 / 03 / 2021
                Adding custom Optimizer Declaration list and make sure that
                registered custom optimizer class has inherit from `Optimizer`.
"""

import numpy as np

class _OptimizerDeclarationDict(dict):
    """
        Custom Dictionary class for Optimizers. It stores registered optimizers.
    """
    def __getitem__(self, key):
        try:
            return self.__dict__[key]
        except KeyError as e:
            msg = f"The Optimizer `{key}` is not registered! " 
            msg += "Please use `REGISTER_OPTIMIZER` method for registering."
            raise KeyError(msg) from None
  
    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

__optimizerDecleration = _OptimizerDeclarationDict()

class Optimizer:
    '''
        Base class of optimizer.
        
        To implement new optimizer method, developer need to declare step function which
        update the weights, biases and relative parameters.

        Containing optimizer methods : \n
            - SGD ('sgd')
            - Adagrad ('adagrad')
            - RMSprop ('rmsprop')
            - AdaDelta ('adadelta')
            - Adam ('adam')

        To call optimizer for training, user have two way. \n
            - User can define optimizer as string in calling NeuralNetwork.setup() function.
                
                Ex: 
                    >>> NeuralNetwork.setup(loss_function='categoricalcrossentropy', optimizer='SGD') \n
                    \n
            - User can define optimizer explicitly.
                
                Ex:
                    >>> opt = gNet.optimizer.SGD(lr=0.001, momentum = 0.5)
                        NeuralNetwork.setup(loss_function='categoricalcrossentropy', optimizer=opt)  

        All class should have step function to calcute new trainable values. 
    '''
    def __init__(self, **kwargs) -> None:
        pass

    def step(self, layers) -> None:
        """
            Evaluation of step w.r.t layers' trainable variables.
        """
        raise NotImplementedError

class SGD(Optimizer):
    '''
        Implementation of SGD.
        SGD is one of the basic optimizer. It has some modification also.
        If you want to use Momentum SGD you can set momentum value.
        If you want Nesterov you can just set it True.
        Bengio Nesterov is used because of automatic differentiation.

        Arguments :
        -----------

        lr           : Learning rate. 
        >>>    type     : float
        >>>    Default  : 0.01
        
        momentum     : Momentum value.
        >>>    type     : float
        >>>    Default  : 0.

        Nesterov     : Nesterov acceleration gradient.
        >>>    type     : bool
        >>>    Default  : False

    '''
    def __init__(self, lr=0.01, momentum=0., Nesterov=False, **kwargs) -> None:
        super(SGD, self).__init__(**kwargs)
        # initializing values.
        self.lr = lr
        self.momentum = momentum
        self.v = []
        self.mids = []
        self.nesterov = Nesterov
        self.init=True

    def step(self, layers)->None:
        # for first time call the step function, initialize parameter w.r.t layer size and trainable variable size.
        if self.init:
            for layer in layers:
                self.v.append(np.zeros_like(layer.trainable))            
            self.init = False
        
        for ind, layer in enumerate(layers):
            if self.nesterov:
                # implementation done from https://jlmelville.github.io/mize/nesterov.html#bengio_formulation
                for ind_tra, trainable in enumerate(layer.trainable):
                    # calculate mid value
                    midVal = self.momentum * self.momentum * self.v[ind][ind_tra]
                    # update trainable value
                    trainable.value += midVal - ( (1 + self.momentum) * self.lr * trainable.grad.value)
                    # calculate next velocity
                    self.v[ind][ind_tra] = self.momentum * self.v[ind][ind_tra] - self.lr * trainable.grad.value
            else:
                for ind_tra, trainable in enumerate(layer.trainable):
                    # calculate next velocity with momentum
                    self.v[ind][ind_tra] = self.momentum * self.v[ind][ind_tra] - self.lr * trainable.grad.value
                    # update trainable value
                    trainable.value += self.v[ind][ind_tra]


class Adagrad(Optimizer):
    '''
        Implementation of Adagrad (Adaptive Gradient).

        Adagrad changes its learning rate adaptively. 
        
        Learning rate recomended as a default.

        Arguments :
        -----------

        lr           : Learning rate. 
        >>>    type     : float
        >>>    Default  : 0.01
        
        epsilon      : Clip value to get rid of 0 division error.
        >>>    type     : float
        >>>    Default  : 1e-7.

    '''
    def __init__(self, lr=0.01, epsilon=1e-7, **kwargs) -> None:
        super(Adagrad, self).__init__(**kwargs)
        # initializing values.
        self.lr = lr
        self.cache = []
        self.eps = epsilon
        self.init = True

    def step(self, layers)->None:
        # for first time call the step function, initialize parameter w.r.t layer size and trainable variable size.
        if self.init:
            for layer in layers:
                self.cache.append(np.zeros_like(layer.trainable))
            self.init = False

        for ind, layer in enumerate(layers):
            for ind_tra, trainable in enumerate(layer.trainable):
                # store square of gradient values of trainable parameter
                self.cache[ind][ind_tra] += trainable.grad.value ** 2
                # update trainable values 
                trainable.value -= self.lr / (np.sqrt(self.cache[ind][ind_tra] + self.eps)) \
                        * trainable.grad.value


class RMSprop(Optimizer):
    '''
        Implementation of RMSprop (Root Mean Squared prop).

        RMSprop changes its learning rate adaptively like Adagrad. 
        It uses different approach.
        
        Beta recomended as a default. Learning rate can be modified if you want.

        Arguments :
        -----------

        lr           : Learning rate. 
        >>>    type     : float
        >>>    Default  : 0.01

        beta         : Update coefficient.
        >>>    type     : float
        >>>    Default  : 0.9
        
        epsilon      : Clip value to get rid of 0 division error.
        >>>    type     : float
        >>>    Default  : 1e-7.

    '''
    def __init__(self, lr=0.001, beta= 0.9, epsilon=1e-7, **kwargs) -> None:
        super(RMSprop, self).__init__(**kwargs)
        # initializing values.
        self.lr = lr
        self.beta = beta
        self.cache = []
        self.eps = epsilon
        self.init = True

    def step(self, layers)->None:
        # for first time call the step function, initialize parameter w.r.t layer size and trainable variable size.
        if self.init:
            for layer in layers:
                self.cache.append(np.zeros_like(layer.trainable))
            self.init = False

        for ind, layer in enumerate(layers):
            for ind_tra, trainable in enumerate(layer.trainable):
                # update cache values 
                self.cache[ind][ind_tra] = self.beta * self.cache[ind][ind_tra] \
                    + (1. - self.beta) * trainable.grad.value ** 2
                # update trainable values 
                trainable.value -= self.lr / (np.sqrt(self.cache[ind][ind_tra] + self.eps)) \
                    * trainable.grad.value


class AdaDelta(Optimizer):
    '''
        Implementation of AdaDelta (Adaptive Delta).

        AdaDelta has no learning rate.
        It uses different from of Adagrad and RMSprop.
        
        Beta recomended as a default.

        Arguments :
        -----------

        beta         : Update coefficient.
        >>>    type     : float
        >>>    Default  : 0.9
        
        epsilon      : Clip value to get rid of 0 division error.
        >>>    type     : float
        >>>    Default  : 1e-7.

    '''
    def __init__(self, beta= 0.9, epsilon=1e-7, **kwargs) -> None:
        super(AdaDelta, self).__init__(**kwargs)
        # initializing values.
        self.D = []
        self.beta = beta
        self.cache = []
        self.delta = []
        self.eps = epsilon
        self.init = True
    
    def step(self, layers)->None:
        # for first time call the step function, initialize parameter w.r.t layer size and trainable variable size.
        if self.init:
            for layer in layers:
                self.cache.append(np.zeros_like(layer.trainable))
                self.delta.append(np.zeros_like(layer.trainable))
                self.D.append(np.zeros_like(layer.trainable))
            self.init = False

        for ind, layer in enumerate(layers):
            for ind_tra, trainable in enumerate(layer.trainable):
                # update cache variable
                self.cache[ind][ind_tra] = self.beta * self.cache[ind][ind_tra] + (1. - self.beta) * trainable.grad.value ** 2
                # update past trainable variables
                self.delta[ind][ind_tra] = (np.sqrt(self.D[ind][ind_tra] + self.eps)) / (np.sqrt(self.cache[ind][ind_tra] + self.eps)) \
                    * trainable.grad.value
                # update D w.r.t trainables and past trainables
                self.D[ind][ind_tra] = self.beta * self.D[ind][ind_tra] + (1. - self.beta) * self.delta[ind][ind_tra] ** 2
                # update past trainables as current one
                trainable.value -= self.delta[ind][ind_tra]


class Adam(Optimizer):
    '''
        Implementation of Adam.

        It uses different from of Adagrad, Adadelta and RMSprop.
        
        Arguments recomended as a default.
        Learning rate can be modified.

        Arguments :
        -----------
        
        lr         : Learning rate.
        >>>    type     : float
        >>>    Default  : 0.001

        beta1        : Update coefficient.
        >>>    type     : float
        >>>    Default  : 0.9

        beta2        : Update coefficient.
        >>>    type     : float
        >>>    Default  : 0.999
        
        epsilon      : Clip value to get rid of 0 division error.
        >>>    type     : float
        >>>   Default  : 1e-7.

    '''
    def __init__(self,lr=0.001, beta1= 0.9, beta2= 0.999, epsilon=1e-7, **kwargs) -> None:
        super(Adam, self).__init__(**kwargs)
        # initializing values.
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.t = 1
        self.init = True
        self.m = []
        self.v = []
        self.mhat = []
        self.vhat = []

    def step(self, layers)->None:
        # for first time call the step function, initialize parameter w.r.t layer size and trainable variable size.
        if self.init:
            for layer in layers:
                self.m.append(np.zeros_like(layer.trainable))
                self.v.append(np.zeros_like(layer.trainable))
                self.mhat.append(np.zeros_like(layer.trainable))
                self.vhat.append(np.zeros_like(layer.trainable))
            self.init = False

        for ind, layer in enumerate(layers):
            for ind_tra, trainable in enumerate(layer.trainable):    
                # update momentum
                self.m[ind][ind_tra] = self.beta1 * self.m[ind][ind_tra] + (1 - self.beta1) * trainable.grad.value
                # update velocity
                self.v[ind][ind_tra] = self.beta2 * self.v[ind][ind_tra] + (1 - self.beta2) * (trainable.grad.value ** 2)
                # update momentum hat
                self.mhat[ind][ind_tra] = self.m[ind][ind_tra] / (1 - self.beta1 ** self.t)
                # update velocity hat
                self.vhat[ind][ind_tra] = self.v[ind][ind_tra] / (1 - self.beta2 ** self.t)
                # update weights and biases
                trainable.value -= self.lr * self.mhat[ind][ind_tra] / (np.sqrt(self.vhat[ind][ind_tra]) + self.eps)
            
        self.t += 1


def REGISTER_OPTIMIZER(optimizer : Optimizer, call_name : str):
    """
        Register Optimizer w.r.t `call_name`. 

        Arguments :
        -----------

        optimizer       : Optimizer class.
        >>>    type     : gNet.optimizer.Optimizer()

        call_name       : Calling name of optimizer. It will be lowercase. It is not sensible.
        >>>    type     : str
        
    """
    if isinstance(optimizer(), Optimizer):
        __optimizerDecleration.update({call_name.lower() : optimizer})
    else:
        msg = f"The Optimizer `{optimizer}` is not same as gNet's Optimizer base class. " 
        msg += f"\nPlease make sure that `{optimizer}` inherit from gNet.optimizer.Optimizer"
        raise TypeError(msg)
    

REGISTER_OPTIMIZER(SGD, 'sgd')
REGISTER_OPTIMIZER(Adagrad, 'adagrad')
REGISTER_OPTIMIZER(RMSprop, 'rmsprop')
REGISTER_OPTIMIZER(AdaDelta, 'adadelta')
REGISTER_OPTIMIZER(Adam, 'adam')