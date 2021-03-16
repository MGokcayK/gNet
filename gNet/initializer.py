"""
    Initializer module of gNet.

    Containing initializer methods (with calling string) : \n
        - Ones init ('ones_init')
        - Zeros init ('zeros_init')
        - He's normal ('he_normal')
        - He's uniform ('he_uniform')
        - Normal init ('normal_init')
        - Uniform init ('uniform_init')
        - Xavier's normal ('xavier_normal')
        - Xavier's uniform ('xavier_uniform')
        - Orthogonal ('orthogonal')

    To call initializer for training, user have one way. \n
        User can declared it in layer declaration. 

        Ex : 
            ...
        >>> initer = gNet.initializer.He_normal()
            net = NeuralNetwork()
            net.add(Dense(100, input_shape=(100,100), initialize_method=initer))
                ...

    Author : @MGokcayK 
    Create : 30 / 03 / 2020
    Update : 16 / 03 / 2021
                Adding custom Initializer Declaration list and make sure that
                registered custom Initializer class has inherit from `Initializer`.
"""

import numpy as np

class _InitializerDeclarationDict(dict):
    """
        Custom Dictionary class for Initializers. It stores registered initializers.
    """
    def __getitem__(self, key):
        try:
            return self.__dict__[key]
        except KeyError as e:
            msg = f"The Initializer `{key}` is not registered! " 
            msg += "Please use `REGISTER_INITIALIZER` method for registering."
            raise KeyError(msg) from None
  
    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

__initializeDeclaretion = _InitializerDeclarationDict()

class Initializer:
    """
        Containing initializer methods (with calling string) : \n
            - Ones init ('ones_init')
            - Zeros init ('zeros_init')
            - He's normal ('he_normal')
            - He's uniform ('he_uniform')
            - Normal init ('normal_init')
            - Uniform init ('uniform_init')
            - Xavier's normal ('xavier_normal')
            - Xavier's uniform ('xavier_uniform')
            - Orthogonal ('orthogonal')

        To call initializer for training, user have one way. \n
            User can declared it in layer declaration. 

            Ex : 
                ...
            >>> initer = gNet.initializer.He_normal()
                net = NeuralNetwork(...)
                net.add(Dense(100, input_shape=(100,100), initialize_method=initer))
                ...


        Initializer class should have get_init methods. This methods return proper 
        output of initializer results. If get_init methods it not added into class, 
        class will not accepted. 

        Also, class can be written in __initializeDeclaretion dict to call with its
        calling strings like 'he_normal' for He_normal class.
        """
    def __init__(self, **kwargs) -> None:
        pass

    def _get_fans(self, shape):
        assert type(shape) == list or type(shape) == tuple, \
            'Please sure that initializer shape is list or tuple.'
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out

    def get_init(self, shape=None, **kwargs) -> np.ndarray:
        raise NotImplementedError


class Zeros_init(Initializer):
    """
        Zeros init is initialize the parameters w.r.t arguments.

        Arguments :
        -----------

            shape   : Shape or zeros. It can be int, list, tuple. 

        Ex: we want 10x5 zeros init matrix.

        >>>  mtrx = Zeros_init().get_init(shape=(10,5)) or Zeros_init().get_init((10,5))
    """ 
    def get_init(self, shape=None, **kwargs) -> np.ndarray: 
        
        return np.zeros(shape=shape)

class Ones_init(Initializer):
    """
        Ones init is initialize the parameters w.r.t arguments.

        Arguments :
        -----------

            shape   : Shape or ones. It can be int, list, tuple. 

        Ex: we want 10x5 ones init matrix.

        >>>  mtrx = Ones_init().get_init(shape=(10,5)) or Ones_init().get_init((10,5))
    """ 
    def get_init(self, shape=None, **kwargs) -> np.ndarray: 
        
        return np.ones(shape=shape)

class Normal_init(Initializer):
    """
        Normal init is initialize the parameters w.r.t arguments.
        Normal init is normal distribution around 0.

        Arguments :
        -----------

            shape       : Shape or normal values.

        
            mean        : Mean of normal distribution.
                Default : 0.
        
            stdDev      : Standart deviation of normal distribution.
                Default : 1.

        Ex: we want 10x5 normal init matrix.

        >>>  mtrx = Normal_init().get_init(shape=(10,5)) or Normal_init().get_init((10,5))
    """
    def __init__(self, mean=0., stdDev=1., **kwargs):
        self._mean = mean
        self._stdDev = stdDev

    def get_init(self, shape=None, **kwargs) -> np.ndarray: 
    
        return np.random.normal(loc=self._mean, scale=self._stdDev, size=shape)

class Uniform_init(Initializer):
    """
        Uniform init is initialize the parameters w.r.t arguments.
        Uniform init is uniform distribution from -0.05 to 0.05.

        Arguments :
        -----------

            shape       : Shape of uniform_init.

            scale       : Scaling coefficient. 
                Default : 0.05

            minVal      : Minimum value of uniform distribution. If not defined, it is equal to -scale.
                Default : -0.05.

            maxVal      : Maximum value of uniform distribution. If not defined, it is equal to scale.
                Default : 0.05.

        Ex: we want 10x5 uniform init matrix.

        >>>  mtrx = Uniform_init().get_init(shape=(10,5)) or Uniform_init().get_init((10,5))
    """ 
    def __init__(self, scale=0.05, minVal=None, maxVal=None, **kwargs):
        self._min = minVal
        self._max = maxVal

        if (self._min == None) or (self._max == None):
            self._min = -scale
            self._max = scale

    def get_init(self, shape=None, **kwargs) -> np.ndarray:   
        
        return np.random.uniform(self._min, self._max, size=shape)
        
class Xavier_normal(Initializer):
    """
        Reference: Glorot & Bengio, AISTATS 2010

        Xavier_normal Xavier Glorot & Bengio's initialize method with normal distribution.
        
        Arguments :
        -----------

            shape   : Shape of Xavier_normal.            
            
        Ex: we want 10x5 Xavier_normal matrix.

        >>>  mtrx = Xavier_normal().get_init(shape=(10,5)) or Xavier_normal().get_init((10,5))
    """
    def get_init(self, shape=None, **kwargs) -> np.ndarray: 

        fan_in, fan_out = self._get_fans(shape)

        s = np.sqrt(2. / (fan_in + fan_out))
        
        return np.random.normal(loc=0.0, scale=s, size=shape)
        
class Xavier_uniform(Initializer):
    """
    Xavier_uniform Xavier Glorot & Bengio's initialize method with uniform distribution.
    
    Arguments :
    -----------

        shape   : Shape of Xavier_uniform.            
        
    Ex: we want 10x5 X Xavier_uniform matrix.

    >>>  mtrx = Xavier_uniform().get_init(shape=(10,5)) or Xavier_uniform().get_init((10,5))
    """
    def get_init(self, shape=None, **kwargs) -> np.ndarray: 

        fan_in, fan_out = self._get_fans(shape)

        s = np.sqrt(6. / (fan_in + fan_out))
        
        return np.random.uniform(-s, s, size=shape)        

class He_normal(Initializer):
    """
        Reference:  He et al., http://arxiv.org/abs/1502.01852

        He_normal He's initialize method with normal distribution.
        
        Arguments :
        -----------

            shape   : Shape of He_normal.            
            
        Ex: we want 10x5 He_normal matrix.

        >>>  mtrx = He_normal().get_init(shape=(10,5)) or He_normal().get_init((10,5))
    """   
    def get_init(self, shape=None, **kwargs) -> np.ndarray: 

        fan_in, fan_out = self._get_fans(shape)

        s = np.sqrt(2. / fan_in)
        
        return np.random.normal(loc=0.0, scale=s, size=shape)

class He_uniform(Initializer):
    """
    Reference:  He et al., http://arxiv.org/abs/1502.01852

    He_normal He's initialize method with uniform distribution.
    
    Arguments :
    -----------

        shape   : Shape of He_uniform.            
        
    Ex: we want 10x5 He_uniform matrix.

    >>>  mtrx = He_uniform().get_init(shape=(10,5)) or He_uniform().get_init((10,5))
    """     
    def get_init(self, shape=None, **kwargs) -> np.ndarray: 

        fan_in, fan_out = self._get_fans(shape)

        s = np.sqrt(6. / fan_in)
        
        return np.random.uniform(-s, s, size=shape)

class Orthogonal(Initializer):
    """
    Orthogonal initialization.
    
    Arguments :
    -----------

        shape   : Shape of orthogonal init.               
        
    Ex: we want 10x5 orthogonal matrix.

    >>>  mtrx = Orthogonal().get_init(shape=(10,5)) or Orthogonal().get_init((10,5))
    """     
    def get_init(self, shape=None, **kwargs) -> np.ndarray: 

        fan_in, fan_out = self._get_fans(shape)
        
        X = np.random.normal(size=shape)

        _, _, Vt = np.linalg.svd(X, full_matrices=False)

        return Vt.reshape(shape)
        

def REGISTER_INITIALIZER(initializer : Initializer, call_name : str):
    """
        Register Initializer w.r.t `call_name`. 

        Arguments :
        -----------

        initializer     : Initializer class.
        >>>    type     : gNet.initializer.Initializer()

        call_name       : Calling name of initializer. It will be lowercase. It is not sensible.
        >>>    type     : str
    """
    if isinstance(initializer(), Initializer):
        __initializeDeclaretion.update({call_name.lower() : initializer})
    else:
        msg = f"The Initializer `{initializer}` is not same as gNet's Initializer base class. " 
        msg += f"\nPlease make sure that `{initializer}` inherit from gNet.initializer.Initializer"
        raise TypeError(msg)
    

REGISTER_INITIALIZER(He_normal, 'he_normal')
REGISTER_INITIALIZER(He_uniform, 'he_uniform')
REGISTER_INITIALIZER(Ones_init, 'ones_init')
REGISTER_INITIALIZER(Zeros_init, 'zeros_init')
REGISTER_INITIALIZER(Normal_init, 'normal_init')
REGISTER_INITIALIZER(Uniform_init, 'uniform_init')
REGISTER_INITIALIZER(Xavier_normal, 'xavier_normal')
REGISTER_INITIALIZER(Xavier_uniform, 'xavier_uniform')
REGISTER_INITIALIZER(Orthogonal, 'orthogonal')