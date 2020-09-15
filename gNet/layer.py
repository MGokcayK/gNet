"""
    Layer module of gNet.

    Containing layers : \n
        - Dense
        - Flatten
        - Activation
        - Conv1D
        - Conv2D
        - Conv3D
        - MaxPool1D
        - MaxPool2D
        - MaxPool3D
        - AveragePool1D
        - AveragePool2D
        - AveragePool3D
        - Dropout
        - Batch Normalization
        - SimpleRNN
        - LSTM
        - GRU

    Layer should be used with Model Class's add method. Rest of the calculation should be done 
    by NN structure.

    Author : @MGokcayK 
    Create : 04 / 04 / 2020
    Update : 15 / 09 / 2020
                Altering 2D Matrix mul. operations from `dot` to `matmul` tensor ops.
"""

# import required modules
import numpy as np
from gNet import tensor as T
from gNet.activation_functions import __activationFunctionsDecleration as AD
from gNet.initializer import __initializeDeclaretion as ID
import gNet.conv_utils as conv_util





class Layer:
    """
        Base class of layer implementation.

        Layer shoudl have two methods which are `_init_trainable`, `__call__` and `compute`.
        These methods can be adapted for proper implementation. 
        `_init_trainable` method can be pass  because of layer's need of initialization.
        `__call__` method should be implemented for each layer to update model parameters.
        `compute` method should be implemented for each layer. Calculation of layer
        done by `compute` method. 

        Base class has also different methods to create easy child class. `_set_initializer`,
        `_get_inits`, and `zero_grad` are helper methods and not called separately by child
        classed. 
    """
    def __init__(self, **kwargs) -> None:
        self._actFuncCaller = AD
        self._trainable = []

    def _set_initializer(self):
        """
            Set initializer methods for layer parameters. If one of the parameters of layer
            have initializer, this function should be called at the end of layer's `__init__` 
            method. 

            This method also have 'bias initializer' which can initialize bias separately. 

            Initialization method can be called in two way. First way is calling by methods
            string which can be found on `initalizer` module or docs. Second way is calling 
            by custom initialization method which can create by base class of initialize 
            module. Descriptions can be found related modules or docs. 
        """
        assert self._initialize_method != None,\
            'Initializer method not be None. Declare with string or explicitly.'

        assert self._bias_initialize_method != None, \
            'Bias initializer method not be None. Declare with string or explicitly.'

        # initializer generic
        if isinstance(self._initialize_method, str):
            _init = self._initialize_method.lower()
            self._initializer = ID[_init]()
        else:
            self._initializer = self._initialize_method
        # bias initializer
        if isinstance(self._bias_initialize_method, str):
            _init = self._bias_initialize_method.lower()
            self._bias_initializer = ID[_init]()
        else:
            self._bias_initializer = self._bias_initialize_method

    def _get_inits(self, W_shape=None, B_shape=None)-> T.Tensor:
        """
                This function called after initializer setted. 

                Function have 2 argumest which pass shape of parameters. 

                Function returns initialized W and B respectively.
        """
        _W = self._initializer.get_init(shape=W_shape)
        if self._bias_initialize_method != None:
            _b = self._bias_initializer.get_init(shape=B_shape)
        else:
            _b = self._initializer.get_init(shape=B_shape)
        return _W, _b

    def _init_trainable(self):
        '''
            Even layer has no trainable parameter, class should has _init_trainable
            method and just `pass`.

            Initialization parameters depends on layer. Thus, implementation should 
            be carefully selected and method should be called in `__call__` method.
        '''
        raise NotImplementedError

    def zero_grad(self):
        """
            This method make zero of all trainable paramters' grad values. 
            This is required for each batch. 
        """
        for trainable in self._trainable:
            trainable.zero_grad()

    def __call__(self, params) -> None:
        """
            `__call__` method is one of the important methods of layer class.
            After adding layer in the Model class, this class called to update
            model parameters which is argued as dict as named `params`. 

            Updated parameters of model can be changed by layer. Thus, be carefull
            for updated parameters. 
        """
        raise NotImplementedError     

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        """
            Compute method is base of computation of layer. This is the core of 
            computation. Implementation should be carefully done. 
            Without compute method, layer cannot be called by NN. 
        """
        raise NotImplementedError

    def regularize(self) -> T.Tensor:
        """
            Regularize method is base of computation of regularization of layer. 
            If regularization is not need in that layer like dropout or flatten, 
            return zero. Implementation should be carefully done. 
            Without regularize method, layer cannot be called by NN. 
        """
        raise NotImplementedError

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value



class Dense(Layer):
    """
        Dense layer implementation. Dense layer is also known as Fully Connected Layer.
        Dense layer is one of the basic layer of neural networks.

        If input of layer is not 1D, it should be flattened before feed the layer.

        Its trainable variable size depends on previous layer's neuron number and current
        layer's neuron number.

        Arguments for initialization :
        ------------------------------

        neuron_number       : Number of neuron of layer.
            >>> type        : int
            >>> Default     : None

        activation_function : Activation function of layer.
            >>> type        : str or custom activation class
            >>> Default     : None

        initialize_method   : Layer initialization method.
            >>> type        : str or custom initializer class
            >>> Default     : 'xavier_uniform'

        bias_initializer    : Layer's bias's initialization method.
            >>> type        : str or custom initializer class
            >>> Default     : 'zeros_init'

        weight_regularizer  : Regularizer method of weights of layer.
            >>> type        : regularizer class
            >>> Default     

        bias_regularizer    : Regularizer method of biases of layer.
            >>> type        : regularizer class
            >>> Default     : None 

        Arguments for compute method is tensor of previous method in proper size.

        Its compute method calculation based on :
        \n\t\t    z = layer_input @ weights + biases
        \n\t\t    layer_output = activation_function(z)
    """
    def __init__(self,
                neuron_number = None,
                activation_function = None,
                initialize_method = 'xavier_uniform',
                bias_initializer = 'zeros_init',
                weight_regularizer = None,
                bias_regularizer = None,
                **kwargs):
        super(Dense, self).__init__(**kwargs)
        if activation_function == None:
            activation_function = 'none'
        self._activation = activation_function
        self._neuronNumber = neuron_number
        self._initialize_method = initialize_method
        self._bias_initialize_method = bias_initializer
        self._set_initializer()
        self._weight_regularizer = weight_regularizer
        self._bias_regularizer = bias_regularizer

    def __call__(self, params) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        self._thisLayer = params['layer_number']
        params['layer_name'].append('Dense : ' + str(self._activation))
        params['activation'].append(self._activation)
        params['model_neuron'].append(self._neuronNumber)
        params['layer_output_shape'].append(self._neuronNumber)

        # if activation function is `str` create caller.
        if isinstance(self._activation, str):
            self._actCaller = self._actFuncCaller[self._activation]()
        else:
            self._actCaller = self._activation

        self._init_trainable(params)

    def _init_trainable(self, params):
        '''
            Initialization of dense layer's trainable variables.
        '''
        if self._thisLayer == 0:
            row = 1
            col = 1
        else:
            row = params['model_neuron'][self._thisLayer-1]
            col = params['model_neuron'][self._thisLayer]

        _w_shape = (row, col)
        _b_shape = [col]

        params['#parameters'].append(row*col+col)

        # get initialized values of weight and biases
        _w, _b = self._get_inits(_w_shape, _b_shape)
        # append weight and biases into trainable as `tensor`.
        self._trainable.append(T.Tensor(_w.astype(np.float32), have_grad=True))
        self._trainable.append(T.Tensor(_b.astype(np.float32), have_grad=True))

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of dense layer.
        '''
        _z_layer = inputs @ self._trainable[0] + self._trainable[1]
        return self._actCaller.activate(_z_layer)

    def regularize(self) -> T.Tensor:
        """
            Regularization of layer. 
        """
        _res = T.Tensor(0.)
        if self._weight_regularizer:
            _res += self._weight_regularizer.compute(self._trainable[0])
        if self._bias_regularizer:
            _res += self._bias_regularizer.compute(self._trainable[1])
        return _res



class Flatten(Layer):
    """
        Flatten layer implementation. Flatten layer is one of the basic layer of neural networks.
        Flatten layer can be in two different conditions which are first layer of model and middle
        layer of model.

        When Flatten layer is first layer of model, input_shape should be declared because layer
        does not know what is the shape of input.

        When Flatten layer is middle layer (it can be anywhere but not first layer) of model,
        input_shape should NOT be declared because layer does know what is shape of input from
        previous layer output parameters in model's params.

        Arguments for initialization :
        ------------------------------

        input_shape      : Shape of input when flatten layer is the first layer of model.
                        Shape will be in form of (channel, height, width).
            >>> type     : tuple
            >>> Default  : None

        Its compute method calculation based on calling flatten operator of tensor.
    """
    def __init__(self, input_shape=None, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self._input_shape = input_shape

    def __call__(self, params) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        self._thisLayer = params['layer_number']
        params['layer_name'].append('flatten')
        params['activation'].append('none')
        params['#parameters'].append(0)

        self._neuronNumber = 1
        # If flatten layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            # calculate flattened layer neuron number
            for item in self._input_shape:
                self._neuronNumber *= item
            self._output_shape = self._neuronNumber

        else:
            # we should calculate shape of input. we should know previous layer output shape
            self._neuronNumber = 1
            assert self._thisLayer != 0, 'Please make sure that flatten layer is not first layer. \n\
            If it is first layer, give it input_shape.'
            for item in params['layer_output_shape'][self._thisLayer - 1]:
                self._neuronNumber *= item
            self._output_shape = self._neuronNumber

        # add output shape to model params
        params['layer_output_shape'].append(self._output_shape)
        # add flattened layer neuron number to model params
        params['model_neuron'].append(self._neuronNumber)

    def _init_trainable(self, params):
        # flatten layer has no initialized parameters. Thus, pass it.
        pass

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of flatten layer.
        '''
        return T.flatten(inputs, batching=True)

    def regularize(self) -> T.Tensor:
        """
            Regularization of layer. This layer do not have regularizable parameter.
        """
        return T.Tensor(0.)



class Activation(Layer):
    """
        Activation layer implementation. Activate the previous layer outputs.

        Arguments for initialization :
        ------------------------------

        activation_function     : Activation function of layer.
            >>> type            : str or custom activation function
            >>> Default         : None
    """
    def __init__(self, activation_function=None, **kwargs):
        super(Activation, self).__init__(**kwargs)
        if activation_function == None:
            activation_function = 'none'
        self._activation = activation_function

    def __call__(self, params) -> None:
        """
            Update some model and class parameters.
        """
        self._thisLayer = params['layer_number']
        params['layer_name'].append('Activation Layer :' + str(self._activation))
        params['activation'].append(self._activation)
        params['#parameters'].append(0)
        params['model_neuron'].append(params['model_neuron'][self._thisLayer - 1])
        params['layer_output_shape'].append(params['layer_output_shape'][self._thisLayer - 1])

        # if activation function is `str` create caller.
        if isinstance(self._activation, str):
            self._actCaller = self._actFuncCaller[self._activation]()
        else:
            self._actCaller = self._activation

    def _init_trainable(self, params):
        # Activation layer has no initialized parameters. Thus, pass it.
        pass

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of activation layer.
        '''
        return self._actCaller.activate(inputs)

    def regularize(self) -> T.Tensor:
        """
            Regularization of layer. This layer do not have regularizable parameter.
        """
        return T.Tensor(0.)



class Conv1D(Layer):
    """
        Conv1D layer implementation. Conv2D implementation done by myself. 
        Results compared with Tensorflows' `tf.nn.conv1D` operation. Same input and
        kernel (difference is channel order) gives equal output with corresponding
        channel order.

        Note:
        -----
        gNet use `channel first` approach. Therefore, make sure that your data have `channel first` shape.

        Arguments for initialization :
        ------------------------------

        filter              : Number of filter.
            >>> type        : int
            >>> Default     : 1

        kernel              : Size of kernel (Width). It should declared seperately.
            >>> type        : int
            >>> Default     : 1

        stride              : Stride of kernel (Height, Width). It should declared seperately.
            >>> type        : int
            >>> Default     : 1

        padding             : How padding applied. 'same' and 'valid' is accepted padding types. 'valid' 
            is not apply padding. 'same' is applying paddiing to make output same size w.r.t `Width`. 
            >>> type        : string
            >>> Default     : 'valid'

        kernel_initializer  : Layer's kernel initialization method.
            >>> type        : str or custom initializer class
            >>> Default     : 'xavier_uniform'

        bias_initializer    : Layer's bias's initialization method.
            >>> type        : str or custom initializer class
            >>> Default     : 'zeros_init'

        kernel_regularizer  : Regularizer method of kernels of layer.
            >>> type        : regularizer class
            >>> Default     

        bias_regularizer    : Regularizer method of biases of layer.
            >>> type        : regularizer class
            >>> Default     : None

        bias                : Bool of using bias during calculation.
            >>> type        : bool
            >>> Default     : True

        input_shape         : If Conv1D is first layer of model, input_shape should be declared.
                            Shape will be in form of (channel, width).
            >>> type        : tuple
            >>> Default     : None

        Arguments for compute method is tensor of previous method in proper size.

        Its compute method calculation based on flatten local space of input and kernels then 
        stored as 2D array. After making 2D array, by using dot product, calculation of all 
        convolution can be done. Then, reshaping result to proper size. 
    """
    def __init__(self,
                filter = 1,
                kernel = 1,
                stride = 1,
                padding = 'valid',
                kernel_initializer = 'xavier_uniform',
                bias_initializer = 'zeros_init',
                kernel_regularizer = None,
                bias_regularizer = None,
                use_bias = True,
                input_shape = None,
                **kwargs):
        super(Conv1D, self).__init__(**kwargs)
        assert np.array(filter).size == 1, 'Make sure that filter size of Conv1D has 1 dimension such as 32, get : ' + str(filter)
        assert np.array(kernel).size == 1, 'Make sure that kernel size of Conv1D has 1 dimension such as 2, get : ' + str(kernel)
        assert np.array(stride).size == 1, 'Make sure that stride size of Conv1D has 1 dimension such as 2, get : ' + str(stride)
        self._input_shape = input_shape
        self._filter = filter
        self._K = kernel
        self._stride = stride
        self._padding = padding.lower()
        self._bias = use_bias
        self._initialize_method = kernel_initializer
        self._bias_initialize_method = bias_initializer
        self._set_initializer()
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer

    def __call__(self, params) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        self._thisLayer = params['layer_number']
        params['layer_name'].append('Conv1D')
        params['activation'].append('none')
        params['#parameters'].append(self._filter * self._K + self._filter)

        # If Conv1D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            assert len(self._input_shape) == 2, 'Make sure that input of Conv1D has 2 dimension without batch such as (1,28).'
            # get channel, width and height of data
            self._C, self._W = self._input_shape
        else:
            # Conv1D layer is not first layer. So get channel and width 
            # of data from previous layer output shape
            assert self._thisLayer != 0, 'First layer of Conv1D should have input_shape!'
            self._C, self._W = params['layer_output_shape'][self._thisLayer - 1]

        if self._padding == 'same':
            # To handle even size kernel padding, we put more constant to right and bottom side 
            # of axis like Tensorflow. 
            # Therefore we need to specify axis's each side padding seperately.
            self._O =  int(np.ceil(self._W / self._stride ))
            pd_w = max((self._O - 1) * self._stride + self._K - self._W, 0) # width padding
            pd_left = int(pd_w / 2) # left side paddding 
            pd_right = int(pd_w - pd_left) # rights side padding 
        else:
            pd_left, pd_right = 0, 0
            self._O =  int((self._W - self._K ) / self._stride + 1)

        self._P = (pd_left, pd_right)

        self._output_shape = (self._filter, self._O)

        # add output shape to model params without batch_size
        params['layer_output_shape'].append(self._output_shape)
        # add flattened layer neuron number to model params
        params['model_neuron'].append(self._filter)

        self._init_trainable(params)

    def _init_trainable(self, params):
        '''
            Initialization of Conv1D layer's trainable variables.
        '''
        # create kernel and bias
        _k_shape = (self._filter, self._C, self._K)
        _b_shape = (self._filter, 1)
        _K, _b = self._get_inits(_k_shape, _b_shape)
        # make kernels as  tensor
        _K = T.Tensor(_K.astype(np.float32), have_grad=True)
        # make biases as tensor
        _b = T.Tensor(_b.astype(np.float32), have_grad=True)
        # add them to trainable list
        self._trainable.append(_K)
        self._trainable.append(_b)

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of Conv1D layer.
        '''
        # getting input shapes separately
        N, C, W = inputs.shape
        # padding 
        if self._padding == 'same':
            inputs.value = np.pad(inputs.value, \
                ((0,0),(0,0),(self._P[0], self._P[1])), \
                    mode='constant', constant_values=0)
        # location of local space
        i = conv_util.get_conv1D_indices(self._K, self._stride, self._O)
        # get local spaces of inputs
        value = inputs[:, :, i].value
        # flat the local spaces
        value = value.transpose(1,3,2,0).reshape(self._K * self._C , -1)
        # dot product of kernels and local spaces
        inputs = T.matmul(T.reshape(self._trainable[0], shape=(self._filter, -1)), T.Tensor(value))
        # adding if use_bias is true
        if self._bias:
            inputs += self._trainable[1]
        # reshape dot product to output shape
        inputs = T.reshape(inputs, (self._filter, self._O, N))
        # arrange dimensions
        inputs = T.transpose(inputs, (2, 0, 1))
        return inputs

    def regularize(self) -> T.Tensor:
        """
            Regularization of layer.
        """
        _res = T.Tensor(0.)
        if self._kernel_regularizer:
            _res += self._kernel_regularizer.compute(self._trainable[0])
        if self._bias_regularizer:
            _res += self._bias_regularizer.compute(self._trainable[1])
        return _res



class Conv2D(Layer):
    """
        Conv2D layer implementation. Conv2D implementation done by im2col methods 
        which can be found on https://cs231n.github.io/convolutional-networks/#overview.

        Note:
        -----
        gNet use `channel first` approach. Therefore, make sure that your data have `channel first` shape.

        Arguments for initialization :
        ------------------------------

        filter              : Number of filter.
            >>> type        : int
            >>> Default     : 1

        kernel              : Size of kernel (Height, Width). It should declared seperately.
            >>> type        : tuple
            >>> Default     : (1,1)

        stride              : Stride of kernel (Height, Width). It should declared seperately.
            >>> type        : tuple
            >>> Default     : (1,1)

        padding             : How padding applied. 'same' and 'valid' is accepted padding types. 'valid' 
            is not apply padding. 'same' is applying paddiing to make output same size w.r.t `Height` and `Width`. 
            >>> type        : string
            >>> Default     : 'valid'

        kernel_initializer  : Layer's kernel initialization method.
            >>> type        : str or custom initializer class
            >>> Default     : 'xavier_uniform'

        bias_initializer    : Layer's bias's initialization method.
            >>> type        : str or custom initializer class
            >>> Default     : 'zeros_init'

        kernel_regularizer  : Regularizer method of kernels of layer.
            >>> type        : regularizer class
            >>> Default     

        bias_regularizer    : Regularizer method of biases of layer.
            >>> type        : regularizer class
            >>> Default     : None

        bias                : Bool of using bias during calculation.
            >>> type        : bool
            >>> Default     : True

        input_shape         : If Conv2D is first layer of model, input_shape should be declared.
                            Shape will be in form of (channel, height, width).
            >>> type        : tuple
            >>> Default     : None

        Arguments for compute method is tensor of previous method in proper size.

        Its compute method calculation based on flatten local space of input and kernels then 
        stored as 2D array. After making 2D array, by using dot product, calculation of all 
        convolution can be done. Then, reshaping result to proper size. 
    """
    def __init__(self,
                filter = 1,
                kernel = (1,1),
                stride = (1,1),
                padding = 'valid',
                kernel_initializer = 'xavier_uniform',
                bias_initializer = 'zeros_init',
                kernel_regularizer = None,
                bias_regularizer = None,
                use_bias = True,
                input_shape = None,
                **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        assert np.array(filter).size == 1, 'Make sure that filter size of Conv2D has 1 dimension such as 32, get : ' + str(filter)
        assert np.array(kernel).size == 2, 'Make sure that kernel size of Conv2D has 2 dimension such as (2,2), get : ' + str(kernel)
        assert np.array(stride).size == 2, 'Make sure that stride size of Conv2D has 2 dimension such as (2,2), get : ' + str(stride)
        self._input_shape = input_shape
        self._filter = filter
        self._HH = kernel[0]
        self._WW = kernel[1]
        self._stride_row = stride[0]
        self._stride_col = stride[1]
        self._padding = padding.lower()
        self._bias = use_bias
        self._initialize_method = kernel_initializer
        self._bias_initialize_method = bias_initializer
        self._set_initializer()
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer

    def __call__(self, params) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        self._thisLayer = params['layer_number']
        params['layer_name'].append('Conv2D')
        params['activation'].append('none')
        params['#parameters'].append(self._filter * self._HH * self._WW + self._filter)

        # If Conv2D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            assert len(self._input_shape) == 3, 'Make sure that input of Conv2D has 3 dimension without batch such as (1,28,28).'
            # get channel, width and height of data
            self._C, self._H, self._W = self._input_shape
        else:
            # Conv2D layer is not first layer. So get channel, width and height
            # of data from previous layer output shape
            assert self._thisLayer != 0, 'First layer of Conv2D should have input_shape!'
            self._C, self._H, self._W = params['layer_output_shape'][self._thisLayer - 1]

        if self._padding == 'same':
            # To handle even size kernel padding, we put more constant to right and bottom side 
            # of axis like Tensorflow. 
            # Therefore we need to specify axis's each side padding seperately.
            self.H_out =  int(np.ceil(self._H / self._stride_row ))
            self.W_out =  int(np.ceil(self._W / self._stride_col ))
            pd_h = max((self.H_out - 1) * self._stride_row + self._HH - self._H, 0) # height padding 
            pd_w = max((self.W_out - 1) * self._stride_col + self._WW - self._W, 0) # width padding
            pd_top = int(pd_h / 2) # top side padding 
            pd_bot = int(pd_h - pd_top) # bottom side padding 
            pd_left = int(pd_w / 2) # left side paddding 
            pd_right = int(pd_w - pd_left) # rights side padding 
        else:
            pd_top, pd_bot, pd_left, pd_right = 0, 0, 0, 0
            self.H_out =  int((self._H - self._HH ) / self._stride_row + 1)
            self.W_out =  int((self._W - self._WW ) / self._stride_col + 1)
            
        self._P = ((pd_top, pd_bot), (pd_left, pd_right))  

        self._output_shape = (self._filter, self.H_out, self.W_out)

        # add output shape to model params without batch_size
        params['layer_output_shape'].append(self._output_shape)
        # add flattened layer neuron number to model params
        params['model_neuron'].append(self._filter)

        #self.fil = open('trainables.txt', 'a')
        self._init_trainable(params)

    def _init_trainable(self, params):
        '''
            Initialization of Conv2D layer's trainable variables.
        '''
        # create kernel and bias
        _k_shape = (self._filter, self._C, self._HH, self._WW)
        _b_shape = (self._filter, 1)
        _K, _b = self._get_inits(_k_shape, _b_shape)
        # make kernels as  tensor
        _K = T.Tensor(_K.astype(np.float32), have_grad=True)
        # make biases as tensor
        _b = T.Tensor(_b.astype(np.float32), have_grad=True)
        # add them to trainable list
        self._trainable.append(_K)
        self._trainable.append(_b)

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of Conv2D layer.
        '''
        # getting input shapes separately
        N, C, H, W = inputs.shape
        # padding 
        if self._padding == 'same':
            inputs.value = np.pad(inputs.value, \
                ((0,0),(0,0),(self._P[0][0], self._P[0][1]),(self._P[1][0], self._P[1][1])), \
                    mode='constant', constant_values=0)
        # location of local space
        k, i, j = conv_util.get_im2col_indices(inputs.shape, (self._HH, self._WW), (self._stride_row, self._stride_col), self._output_shape )
        # get local spaces of inputs
        value = inputs[:, k, i, j].value
        # flat the local spaces
        value = value.transpose(1,2,0).reshape((self._HH * self._WW * C, -1))
        # dot product of kernels and local spaces
        inputs = T.matmul(T.reshape(self._trainable[0], shape=(self._filter, -1)), T.Tensor(value)  )
        # adding if use_bias is true
        if self._bias:
            inputs += self._trainable[1]
        # reshape dot product to output shape
        inputs = T.reshape(inputs, (self._filter, self.H_out, self.W_out, N))
        # arrange dimensions
        inputs = T.transpose(inputs, (3, 0, 1, 2))
        return inputs

    def regularize(self) -> T.Tensor:
        """
            Regularization of layer.
        """
        _res = T.Tensor(0.)
        if self._kernel_regularizer:
            _res += self._kernel_regularizer.compute(self._trainable[0])
        if self._bias_regularizer:
            _res += self._bias_regularizer.compute(self._trainable[1])
        return _res



class Conv3D(Layer):
    """
        Conv3D layer implementation. Conv3D implementation done by @Author based on Conv2D.

        Note:
        -----
        gNet use `channel first` approach. Therefore, make sure that your data have `channel first` shape.

        Arguments for initialization :
        ------------------------------

        filter              : Number of filter.
            >>> type        : int
            >>> Default     : 1

        kernel              : Size of kernel (Height, Width, Depth). It should declared seperately.
            >>> type        : tuple
            >>> Default     : (1,1,1)

        stride              : Stride of kernel (Height, Width, Depth). It should declared seperately.
            >>> type        : tuple
            >>> Default     : (1,1,1)

        padding             : How padding applied. 'same' and 'valid' is accepted padding types. 'valid' 
            is not apply padding. 'same' is applying paddiing to make output same size w.r.t `Height`, `Width`
            and `Depth`. 
            >>> type        : string
            >>> Default     : 'valid'

        kernel_initializer  : Layer's kernel initialization method.
            >>> type        : str or custom initializer class
            >>> Default     : 'xavier_uniform'

        bias_initializer    : Layer's bias's initialization method.
            >>> type        : str or custom initializer class
            >>> Default     : 'zeros_init'

        kernel_regularizer  : Regularizer method of kernels of layer.
            >>> type        : regularizer class
            >>> Default     

        bias_regularizer    : Regularizer method of biases of layer.
            >>> type        : regularizer class
            >>> Default     : None

        bias                : Bool of using bias during calculation.
            >>> type        : bool
            >>> Default     : True

        input_shape         : If Conv3D is first layer of model, input_shape should be declared.
                            Shape will be in form of (channel, depth, height, width).
            >>> type        : tuple
            >>> Default     : None

        Arguments for compute method is tensor of previous method in proper size.

        Its compute method calculation based on flatten local space of input and kernels then 
        stored as 2D array. After making 2D array, by using dot product, calculation of all 
        convolution can be done. Then, reshaping result to proper size. 
    """
    def __init__(self,
                filter = 1,
                kernel = (1,1,1),
                stride = (1,1,1),
                padding = 'valid',
                kernel_initializer = 'xavier_uniform',
                bias_initializer = 'zeros_init',
                kernel_regularizer = None,
                bias_regularizer = None,
                use_bias = True,
                input_shape = None,
                **kwargs):
        super(Conv3D, self).__init__(**kwargs)
        assert np.array(filter).size == 1, 'Make sure that filter size of Conv3D has 1 dimension such as 32, get : ' + str(filter)
        assert np.array(kernel).size == 3, 'Make sure that kernel size of Conv3D has 3 dimension such as (2,2,2), get : ' + str(kernel)
        assert np.array(stride).size == 3, 'Make sure that stride size of Conv3D has 3 dimension such as (2,2,2), get : ' + str(stride)
        self._input_shape = input_shape
        self._filter = filter
        self._K_shape = kernel
        self._stride = stride
        self._padding = padding.lower()
        self._bias = use_bias
        self._initialize_method = kernel_initializer
        self._bias_initialize_method = bias_initializer
        self._set_initializer()
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer

    def __call__(self, params) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        self._thisLayer = params['layer_number']
        params['layer_name'].append('Conv3D')
        params['activation'].append('none')

        # If Conv3D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            assert len(self._input_shape) == 4, 'Make sure that input of Conv3D has 4 dimension without batch such as (3,10,28,28).'
            # get channel, width and height of data
            self._C, self._D, self._H, self._W = self._input_shape
        else:
            # Conv3D layer is not first layer. So get channel, depth, width and height
            # of data from previous layer output shape
            assert self._thisLayer != 0, 'First layer of Conv3D should have input_shape!'
            self._C, self._D, self._H, self._W = params['layer_output_shape'][self._thisLayer - 1]

        params['#parameters'].append(self._filter * self._K_shape[0] * self._K_shape[1] * self._K_shape[2] * self._C + self._filter)

        if self._padding == 'same':
            # To handle even size kernel padding, we put more constant to right and bottom side 
            # of axis like Tensorflow. 
            # Therefore we need to specify axis's each side padding seperately.
            self.H_out =  int(np.ceil(self._H / self._stride[0] ))
            self.W_out =  int(np.ceil(self._W / self._stride[1] ))
            self.D_out =  int(np.ceil(self._D / self._stride[2] ))
            pd_h = max((self.H_out - 1) * self._stride[0] + self._K_shape[0] - self._H, 0) # height padding 
            pd_w = max((self.W_out - 1) * self._stride[1] + self._K_shape[1] - self._W, 0) # width padding
            pd_d = max((self.D_out - 1) * self._stride[2] + self._K_shape[2] - self._D, 0) # depth padding
            pd_top = int(pd_h / 2) # top side padding 
            pd_bot = int(pd_h - pd_top) # bottom side padding 
            pd_left = int(pd_w / 2) # left side paddding 
            pd_right = int(pd_w - pd_left) # rights side padding 
            pd_front = int(pd_d // 2) # front side padding 
            pd_back = int(pd_d - pd_front) # back side padding 
        else:
            pd_top, pd_bot, pd_left, pd_right, pd_front, pd_back = 0, 0, 0, 0, 0, 0 
            self.H_out =  int((self._H - self._K_shape[0] ) / self._stride[0] + 1)
            self.W_out =  int((self._W - self._K_shape[1] ) / self._stride[1] + 1)
            self.D_out =  int((self._D - self._K_shape[2] ) / self._stride[2] + 1)
        
        
        self._P = ((pd_top, pd_bot), (pd_left, pd_right), (pd_front, pd_back))     
        

        self._output_shape = (self._filter, self.D_out, self.H_out, self.W_out)

        
        # add output shape to model params without batch_size
        params['layer_output_shape'].append(self._output_shape)
        # add flattened layer neuron number to model params
        params['model_neuron'].append(self._filter)

        self._init_trainable(params)

    def _init_trainable(self, params):
        '''
            Initialization of Conv3D layer's trainable variables.
        '''
        # create kernel and bias
        _k_shape = (self._filter, self._C, self._K_shape[2], self._K_shape[0], self._K_shape[1])
        _b_shape = (self._filter, 1)
        _K, _b = self._get_inits(_k_shape, _b_shape)
        # make kernels as  tensor
        _K = T.Tensor(_K.astype(np.float32), have_grad=True)
        # make biases as tensor
        _b = T.Tensor(_b.astype(np.float32), have_grad=True)
        # add them to trainable list
        self._trainable.append(_K)
        self._trainable.append(_b)

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of Conv3D layer.
        '''
        # getting input shapes separately
        N, C, D, H, W = inputs.shape
        # base_shape 
        base_shape = inputs.shape
        # padding 
        if self._padding == 'same':
            inputs.value = np.pad(inputs.value, \
                ((0,0),(0,0),(self._P[2][0],self._P[2][1]),(self._P[0][0], self._P[0][1]),(self._P[1][0], self._P[1][1])), \
                    mode='constant', constant_values=0)
        # location of local space
        l, k, i, j = conv_util.get_conv3D_indices(base_shape, self._K_shape, self._stride, self._output_shape)
        # get local spaces of inputs
        value = inputs[:,l, k, i, j].value
        # flat the local spaces
        value = value.transpose(1,2,0).reshape((self._K_shape[0] * self._K_shape[1] * self._K_shape[2] * C, -1))
        # dot product of kernels and local spaces
        inputs = T.matmul(T.reshape(self._trainable[0], shape=(self._filter, -1)), T.Tensor(value)  )
        # adding if use_bias is true
        if self._bias:
            inputs += self._trainable[1]
        # reshape dot product to output shape
        inputs = T.reshape(inputs, (self._filter, self.D_out, self.H_out, self.W_out, N))
        # arrange dimensions
        inputs = T.transpose(inputs, (4, 0, 1, 2, 3))
        return inputs

    def regularize(self) -> T.Tensor:
        """
            Regularization of layer.
        """
        _res = T.Tensor(0.)
        if self._kernel_regularizer:
            _res += self._kernel_regularizer.compute(self._trainable[0])
        if self._bias_regularizer:
            _res += self._bias_regularizer.compute(self._trainable[1])
        return _res



class MaxPool1D(Layer):
    """
        MaxPool1D layer implementation implemented by myself based on MaxPool2D.
        
        MaxPooling is getting most dominant feature of data.

        Arguments for initialization :
        ------------------------------

        kernel              : Size of kernel Width.
            >>> type        : int
            >>> Default     : 2

        stride              : Stride of kernel Width
            >>> type        : int
            >>> Default     : 2
        
        padding             : How padding applied. 'same' and 'valid' is accepted padding types. 'valid' 
            is not apply padding. 'same' is applying paddiing to make output same size w.r.t `Width`. 
            >>> type        : string
            >>> Default     : 'valid'

        input_shape         : If MaxPool1D is first layer of model, input_shape should be declared.
                            Shape will be in form of (channel, width).
            >>> type        : tuple
            >>> Default     : None

        Implementation done by finding max values index and getting them.
    """
    def __init__(self,
                kernel = 2,
                stride = 2,
                padding = 'valid',
                input_shape = None,
                **kwargs):
        super(MaxPool1D, self).__init__(**kwargs)
        assert np.array(kernel).size == 1, 'Make sure that kernel size of MaxPool1D has 1 dimension such as 2, get : ' + str(kernel)
        assert np.array(stride).size == 1, 'Make sure that stride size of MaxPool1D has 1 dimension such as 2, get : ' + str(stride)
        self._input_shape = input_shape
        self._K = kernel
        self._stride = stride
        self._padding = padding.lower()

    def __call__(self, params) -> None:
        self._thisLayer = params['layer_number']
        params['layer_name'].append('MaxPool1D')
        params['activation'].append('none')
        params['#parameters'].append(0)

        # If MaxPool1D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            # get channel and width of data
            self._C, self._W = self._input_shape
        else:
            # MaxPool1D layer is not first layer. So get channel and width 
            # of data from previous layer output shape
            self._C, self._W = params['layer_output_shape'][self._thisLayer - 1]

        if self._padding == 'same':
            # To handle even size kernel padding, we put more constant to right and bottom side 
            # of axis like Tensorflow. 
            # Therefore we need to specify axis's each side padding seperately.
            self._O =  int(np.ceil(self._W / self._stride ))
            pd_w = max((self._O - 1) * self._stride + self._K - self._W, 0) # width padding
            pd_left = int(pd_w / 2) # left side paddding 
            pd_right = int(pd_w - pd_left) # rights side padding 
        else:
            pd_left, pd_right = 0, 0
            self._O =  int((self._W - self._K ) / self._stride + 1)
        
        self._P = (pd_left, pd_right)

        self._output_shape = (self._C, self._O)

        # add output shape to model params without batch_size
        params['layer_output_shape'].append(self._output_shape)
        # add channel number to layer neuron number into model params
        params['model_neuron'].append(self._C)

    def _init_trainable(self, params):
        # MaxPool1D layer has no initialized parameters. Thus, pass it.
        pass

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of MaxPool1D layer.
        '''
        # padding 
        if self._padding == 'same':
            inputs.value = np.pad(inputs.value, \
                ((0,0),(0,0),(self._P[0], self._P[1])), \
                    mode='constant', constant_values=0)
        # gettin input shapes separately
        N, C, W = inputs.shape
        # findin pooling locations
        i = conv_util.get_conv1D_indices(self._K, self._stride, self._O)
        # getting local spaces
        inputs = inputs[:, :, i]
        # arrange dimensions
        inputs = T.transpose(inputs,(3,0,1,2))
        # flat local spaces
        inputs = T.reshape(inputs, (self._K, -1))         
        # find max values' index 
        max_idx = np.argmax(inputs.value, axis=0)
        # get max values
        inputs = inputs[max_idx, range(max_idx.size)] 
        ## reshape it to output
        inputs = T.reshape(inputs, (N, -1, self._O))
        return inputs

    def regularize(self) -> T.Tensor:
        """
            Regularization of layer. This layer do not have regularizable parameter.
        """
        return T.Tensor(0.)
        


class MaxPool2D(Layer):
    """
        MaxPool2D layer implementation which can be found on 
        https://cs231n.github.io/convolutional-networks/#overview.

        MaxPooling is getting most dominant feature of data.

        Arguments for initialization :
        ------------------------------

        kernel              : Size of kernel (Height, Width). It should declared seperately.
            >>> type        : tuple
            >>> Default     : (2,2)

        stride              : Stride of kernel (Height, Width). It should declared seperately.
            >>> type        : tuple
            >>> Default     : (2,2)
        
        padding             : How padding applied. 'same' and 'valid' is accepted padding types. 'valid' 
            is not apply padding. 'same' is applying paddiing to make output same size w.r.t `Height` and `Width`. 
            >>> type        : string
            >>> Default     : 'valid'

        input_shape         : If MaxPool2D is first layer of model, input_shape should be declared.
                            Shape will be in form of (channel, height, width).
            >>> type        : tuple
            >>> Default     : None

        Implementation done by finding max values index and getting them.
    """
    def __init__(self,
                kernel = (2,2),
                stride = (2,2),
                padding = 'valid',
                input_shape = None,
                **kwargs):
        super(MaxPool2D, self).__init__(**kwargs)
        assert np.array(kernel).size == 2, 'Make sure that kernel size of MaxPool2D has 2 dimension such as (2,2), get : ' + str(kernel)
        assert np.array(stride).size == 2, 'Make sure that stride size of MaxPool2D has 2 dimension such as (2,2), get : ' + str(stride)
        self._input_shape = input_shape
        self._HH = kernel[0]
        self._WW = kernel[1]
        self._stride_row = stride[0]
        self._stride_col = stride[1]
        self._padding = padding.lower()

    def __call__(self, params) -> None:
        self._thisLayer = params['layer_number']
        params['layer_name'].append('MaxPool2D')
        params['activation'].append('none')
        params['#parameters'].append(0)

        # If MaxPool2D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            # get channel, width and height of data
            self._C, self._H, self._W = self._input_shape
        else:
            # MaxPool2D layer is not first layer. So get channel, width and height
            # of data from previous layer output shape
            self._C, self._H, self._W = params['layer_output_shape'][self._thisLayer - 1]

        if self._padding == 'same':
            # To handle even size kernel padding, we put more constant to right and bottom side 
            # of axis like Tensorflow. 
            # Therefore we need to specify axis's each side padding seperately.
            self.H_out =  int(np.ceil(self._H / self._stride_row ))
            self.W_out =  int(np.ceil(self._W / self._stride_col ))
            pd_h = max((self.H_out - 1) * self._stride_row + self._HH - self._H, 0) # height padding 
            pd_w = max((self.W_out - 1) * self._stride_col + self._WW - self._W, 0) # width padding
            pd_top = int(pd_h / 2) # top side padding 
            pd_bot = int(pd_h - pd_top) # bottom side padding 
            pd_left = int(pd_w / 2) # left side paddding 
            pd_right = int(pd_w - pd_left) # rights side padding 
        else:
            pd_top, pd_bot, pd_left, pd_right = 0, 0, 0, 0
            self.H_out =  int((self._H - self._HH ) / self._stride_row + 1)
            self.W_out =  int((self._W - self._WW ) / self._stride_col + 1)
            
        self._P = ((pd_top, pd_bot), (pd_left, pd_right))  

        self._output_shape = (self._C, self.H_out, self.W_out)

        # add output shape to model params without batch_size
        params['layer_output_shape'].append(self._output_shape)
        # add channel number to layer neuron number into model params
        params['model_neuron'].append(self._C)

    def _init_trainable(self, params):
        # MaxPool2D layer has no initialized parameters. Thus, pass it.
        pass

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of MaxPool2D layer.
        '''
        # padding 
        if self._padding == 'same':
            inputs.value = np.pad(inputs.value, \
                ((0,0),(0,0),(self._P[0][0], self._P[0][1]),(self._P[1][0], self._P[1][1])), \
                    mode='constant', constant_values=0)
        # gettin input shapes separately
        N, C, H, W = inputs.shape
        # reshape inputs
        inputs = T.reshape(inputs, (N * C, 1, H, W))
        # getting temp input shape
        tmp = inputs.shape
        # findin pooling locations
        k, i, j = conv_util.get_im2col_indices(inputs.shape, (self._HH, self._WW), (self._stride_row, self._stride_col), self._output_shape)
        # getting local spaces
        inputs = inputs[:, k,i,j]
        # arrange dimensions
        inputs = T.transpose(inputs,(1,2,0))
        # flat local spaces
        inputs = T.reshape(inputs, (self._HH * self._WW * tmp[1], -1)) 
        # find max values' index 
        max_idx = np.argmax(inputs.value, axis=0)
        # get max values
        inputs = inputs[max_idx, range(max_idx.size)] 
        # reshape it to output
        inputs = T.reshape(inputs, (self.H_out, self.W_out, N, C))
        # arrange dimensions
        inputs = T.transpose(inputs, (2,3,0,1))
        return inputs

    def regularize(self) -> T.Tensor:
        """
            Regularization of layer. This layer do not have regularizable parameter.
        """
        return T.Tensor(0.)



class MaxPool3D(Layer):
    """
        MaxPool3D layer implemented by @Author based on MaxPool2D.

        MaxPooling is getting most dominant feature of data.

        Arguments for initialization :
        ------------------------------

        kernel              : Size of kernel (Height, Width, Depth). It should declared seperately.
            >>> type        : tuple
            >>> Default     : (2,2,2)

        stride              : Stride of kernel (Height, Width, Depth). It should declared seperately.
            >>> type        : tuple
            >>> Default     : (2,2,2)

        padding             : How padding applied. 'same' and 'valid' is accepted padding types. 'valid' 
            is not apply padding. 'same' is applying paddiing to make output same size w.r.t `Height`, `Width`
            and `Depth`. 
            >>> type        : string
            >>> Default     : 'valid'

        input_shape         : If MaxPool3D is first layer of model, input_shape should be declared.
                            Shape will be in form of (channel, depth, height, width).
            >>> type        : tuple
            >>> Default     : None

        Implementation done by finding max values index and getting them.
    """
    def __init__(self,
                kernel = (2,2,2),
                stride = (2,2,2),
                padding = 'valid',
                input_shape = None,
                **kwargs):
        super(MaxPool3D, self).__init__(**kwargs)
        assert np.array(kernel).size == 3, 'Make sure that kernel size of MaxPool3D has 3 dimension such as (2,2,2), get : ' + str(kernel)
        assert np.array(stride).size == 3, 'Make sure that stride size of MaxPool3D has 3 dimension such as (2,2,2), get : ' + str(stride)
        self._input_shape = input_shape
        self._K_shape = kernel
        self._stride = stride
        self._padding = padding.lower()

    def __call__(self, params) -> None:
        self._thisLayer = params['layer_number']
        params['layer_name'].append('MaxPool3D')
        params['activation'].append('none')
        params['#parameters'].append(0)

        # If MaxPool3D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            # get channel, depth, width and height of data
            self._C, self._D, self._H, self._W = self._input_shape
        else:
            # MaxPool3D layer is not first layer. So get channel, width and height
            # of data from previous layer output shape
            self._C, self._D, self._H, self._W = params['layer_output_shape'][self._thisLayer - 1]

        if self._padding == 'same':
            # To handle even size kernel padding, we put more constant to right and bottom side 
            # of axis like Tensorflow. 
            # Therefore we need to specify axis's each side padding seperately.
            self.H_out =  int(np.ceil(self._H / self._stride[0] ))
            self.W_out =  int(np.ceil(self._W / self._stride[1] ))
            self.D_out =  int(np.ceil(self._D / self._stride[2] ))
            print(self.H_out, self.W_out, self.D_out)
            pd_h = max((self.H_out - 1) * self._stride[0] + self._K_shape[0] - self._H, 0) # height padding 
            pd_w = max((self.W_out - 1) * self._stride[1] + self._K_shape[1] - self._W, 0) # width padding
            pd_d = max((self.D_out - 1) * self._stride[2] + self._K_shape[2] - self._D, 0) # depth padding
            pd_top = int(pd_h / 2) # top side padding 
            pd_bot = int(pd_h - pd_top) # bottom side padding 
            pd_left = int(pd_w / 2) # left side paddding 
            pd_right = int(pd_w - pd_left) # rights side padding 
            pd_front = int(pd_d // 2) # front side padding 
            pd_back = int(pd_d - pd_front) # back side padding 
        else:
            pd_top, pd_bot, pd_left, pd_right, pd_front, pd_back = 0, 0, 0, 0, 0, 0 
            self.H_out =  int((self._H - self._K_shape[0] ) / self._stride[0] + 1)
            self.W_out =  int((self._W - self._K_shape[1] ) / self._stride[1] + 1)
            self.D_out =  int((self._D - self._K_shape[2] ) / self._stride[2] + 1)
        
        
        self._P = ((pd_top, pd_bot), (pd_left, pd_right), (pd_front, pd_back))   

        self._output_shape = (self._C, self.D_out, self.H_out, self.W_out)

        # add output shape to model params without batch_size
        params['layer_output_shape'].append(self._output_shape)
        # add channel number to layer neuron number into model params
        params['model_neuron'].append(self._C)

    def _init_trainable(self, params):
        # MaxPool3D layer has no initialized parameters. Thus, pass it.
        pass

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of MaxPool3D layer.
        '''
        # apply padding
        if self._padding == 'same':
            inputs.value = np.pad(inputs.value, \
                ((0,0),(0,0),(self._P[2][0],self._P[2][1]),(self._P[0][0], self._P[0][1]),(self._P[1][0], self._P[1][1])), \
                    mode='constant', constant_values=0)
        # gettin input shapes separately
        N, C, D, H, W = inputs.shape
        # reshape inputs
        inputs = T.reshape(inputs, (N * C, 1, D, H, W))
        # findin pooling locations
        l, k, i, j = conv_util.get_conv3D_indices(inputs.shape, self._K_shape, self._stride, self._output_shape)
        # getting local spaces
        inputs = inputs[:,l, k,i,j]
        # arrange dimensions
        inputs = T.transpose(inputs,(1,2,0))
        # flat local spaces
        inputs = T.reshape(inputs, (self._K_shape[0] * self._K_shape[1] * self._K_shape[2] , -1)) 
        # find max values' index 
        max_idx = np.argmax(inputs.value, axis=0)
        # get max values
        inputs = inputs[max_idx, range(max_idx.size)] 
        # reshape it to output
        inputs = T.reshape(inputs, (self.D_out, self.H_out, self.W_out, N, C))
        # arrange dimensions
        inputs = T.transpose(inputs, (3,4,0,1,2))
        return inputs

    def regularize(self) -> T.Tensor:
        """
            Regularization of layer. This layer do not have regularizable parameter.
        """
        return T.Tensor(0.)



class AveragePool1D(Layer):
    """
        AveragePool1D layer implementation which done by myself based on AveragePool2D.

        AveragePool is getting average feature of data.

        Arguments for initialization :
        ------------------------------

        kernel                  : Size of kernel Width. 
            >>> type            : int
            >>> Default         : 2

        stride                  : Stride of kernel Width. 
            >>> type            : int
            >>> Default         : 2

        padding             : How padding applied. 'same' and 'valid' is accepted padding types. 'valid' 
            is not apply padding. 'same' is applying paddiing to make output same size w.r.t `Width`. 
            >>> type        : string
            >>> Default     : 'valid'

        input_shape             : If AveragePool1D is first layer of model, input_shape should be declared.
                                Shape will be in form of (channel, width).
            >>> type            : tuple
            >>> Default         : None

        Implementation done by finding average values.
    """
    def __init__(self,
                kernel = 2,
                stride = 2,
                padding = 'valid',
                input_shape = None,
                **kwargs):
        super(AveragePool1D, self).__init__(**kwargs)
        assert np.array(kernel).size == 1, 'Make sure that kernel size of AveragePool1D has 1 dimension such as 2, get : ' + str(kernel)
        assert np.array(stride).size == 1, 'Make sure that stride size of AveragePool1D has 1 dimension such as 2, get : ' + str(stride)
        self._input_shape = input_shape
        self._K = kernel
        self._stride = stride
        self._padding = padding.lower()
        
    def __call__(self, params) -> None:
        self._thisLayer = params['layer_number']
        params['layer_name'].append('AveragePool1D')
        params['activation'].append('none')
        params['#parameters'].append(0)

        # If AveragePool1D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            # get channel and width of data
            self._C, self._W = self._input_shape
        else:
            # AveragePool1D layer is not first layer. So get channel and width 
            # of data from previous layer output shape
            self._C, self._W = params['layer_output_shape'][self._thisLayer - 1]

        if self._padding == 'same':
            # To handle even size kernel padding, we put more constant to right and bottom side 
            # of axis like Tensorflow. 
            # Therefore we need to specify axis's each side padding seperately.
            self._O =  int(np.ceil(self._W / self._stride ))
            pd_w = max((self._O - 1) * self._stride + self._K - self._W, 0) # width padding
            pd_left = int(pd_w / 2) # left side paddding 
            pd_right = int(pd_w - pd_left) # rights side padding 
            print(pd_w, pd_left, pd_right)
        else:
            pd_left, pd_right = 0, 0
            self._O =  int((self._W - self._K ) / self._stride + 1)
        
        self._P = (pd_left, pd_right)
        
        self._output_shape = (self._C, self._O)

        # add output shape to model params without batch_size
        params['layer_output_shape'].append(self._output_shape)
        # add channel number to layer neuron number into model params
        params['model_neuron'].append(self._C)

    def _init_trainable(self, params):
        # AveragePool1D layer has no initialized parameters. Thus, pass it.
        pass

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of AveragePool1D layer.
        '''
        # padding 
        if self._padding == 'same':
            inputs.value = np.pad(inputs.value, \
                ((0,0),(0,0),(self._P[0], self._P[1])), \
                    mode='constant', constant_values=0)
        # gettin input shapes separately
        N, C, W = inputs.shape
        #print(inputs)
        # findin pooling locations
        i = conv_util.get_conv1D_indices(self._K, self._stride, self._O)
        # getting local spaces
        inputs = inputs[:, :, i]
        # arrange dimensions
        inputs = T.transpose(inputs,(3,0,1,2))
        # flat local spaces
        inputs = T.reshape(inputs, (self._K, -1))         
        # get mean values 
        #print(inputs)
        inputs = T.mean(inputs, axis=0)
        #print(inputs)
        # reshape it to output
        inputs = T.reshape(inputs, (N, -1, self._O))
        return inputs
        
    def regularize(self) -> T.Tensor:
        """
            Regularization of layer. This layer do not have regularizable parameter.
        """
        return T.Tensor(0.) 



class AveragePool2D(Layer):
    """
        AveragePool2D layer implementation which can be found on 
        https://cs231n.github.io/convolutional-networks/#overview.

        AveragePool is getting average feature of data.

        Arguments for initialization :
        ------------------------------

        kernel                  : Size of kernel (Height, Width). It should declared seperately.
            >>> type            : tuple
            >>> Default         : (2,2)

        stride                  : Stride of kernel (Height, Width). It should declared seperately.
            >>> type            : tuple
            >>> Default         : (2,2)

        padding             : How padding applied. 'same' and 'valid' is accepted padding types. 'valid' 
            is not apply padding. 'same' is applying paddiing to make output same size w.r.t `Height` and `Width`. 
            >>> type        : string
            >>> Default     : 'valid'

        input_shape             : If AveragePool2D is first layer of model, input_shape should be declared.
                                Shape will be in form of (channel, height, width).
            >>> type            : tuple
            >>> Default         : None

        Implementation done by finding average values.
    """
    def __init__(self,
                kernel = (2,2),
                stride = (2,2),
                padding = 'valid',
                input_shape = None,
                **kwargs):
        super(AveragePool2D, self).__init__(**kwargs)
        assert np.array(kernel).size == 2, 'Make sure that kernel size of AveragePool2D has 2 dimension such as (2,2), get : ' + str(kernel)
        assert np.array(stride).size == 2, 'Make sure that stride size of AveragePool2D has 2 dimension such as (2,2), get : ' + str(stride)
        self._input_shape = input_shape
        self._HH = kernel[0]
        self._WW = kernel[1]
        self._stride_row = stride[0]
        self._stride_col = stride[1]
        self._padding = padding.lower()

    def __call__(self, params) -> None:
        self._thisLayer = params['layer_number']
        params['layer_name'].append('AveragePool2D')
        params['activation'].append('none')
        params['#parameters'].append(0)

        # If AveragePool2D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            # get channel, width and height of data
            self._C, self._H, self._W = self._input_shape
        else:
            # AveragePool2D layer is not first layer. So get channel, width and height
            # of data from previous layer output shape
            self._C, self._H, self._W = params['layer_output_shape'][self._thisLayer - 1]

        if self._padding == 'same':
            # To handle even size kernel padding, we put more constant to right and bottom side 
            # of axis like Tensorflow. 
            # Therefore we need to specify axis's each side padding seperately.
            self.H_out =  int(np.ceil(self._H / self._stride_row ))
            self.W_out =  int(np.ceil(self._W / self._stride_col ))
            pd_h = max((self.H_out - 1) * self._stride_row + self._HH - self._H, 0) # height padding 
            pd_w = max((self.W_out - 1) * self._stride_col + self._WW - self._W, 0) # width padding
            pd_top = int(pd_h / 2) # top side padding 
            pd_bot = int(pd_h - pd_top) # bottom side padding 
            pd_left = int(pd_w / 2) # left side paddding 
            pd_right = int(pd_w - pd_left) # rights side padding 
        else:
            pd_top, pd_bot, pd_left, pd_right = 0, 0, 0, 0
            self.H_out =  int((self._H - self._HH ) / self._stride_row + 1)
            self.W_out =  int((self._W - self._WW ) / self._stride_col + 1)
            
        self._P = ((pd_top, pd_bot), (pd_left, pd_right))  

        self._output_shape = (self._C, self.H_out, self.W_out)

        # add output shape to model params without batch_size
        params['layer_output_shape'].append(self._output_shape)
        # add channel number to layer neuron number into model params
        params['model_neuron'].append(self._C)

    def _init_trainable(self, params):
        # AveragePool2D layer has no initialized parameters. Thus, pass it.
        pass

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of AveragePool2D layer.
        '''
        # padding 
        if self._padding == 'same':
            inputs.value = np.pad(inputs.value, \
                ((0,0),(0,0),(self._P[0][0], self._P[0][1]),(self._P[1][0], self._P[1][1])), \
                    mode='constant', constant_values=0)
        # gettin input shapes separately
        N, C, H, W = inputs.shape
        # reshape inputs
        inputs = T.reshape(inputs, (N * C, 1, H, W))
        # getting temp input shape
        tmp = inputs.shape
        # findin pooling locations
        k, i, j = conv_util.get_im2col_indices(inputs.shape, (self._HH, self._WW), (self._stride_row, self._stride_col), self._output_shape)
        # getting local spaces
        inputs = inputs[:, k,i,j]
        # arrange dimensions
        inputs = T.transpose(inputs,(1,2,0))
        # flat local spaces
        inputs = T.reshape(inputs, (self._HH * self._WW * tmp[1], -1)) 
        # find mean values through axis 0
        inputs = T.mean(inputs, axis=0) 
        # reshape it to output
        inputs = T.reshape(inputs, (self.H_out, self.W_out, N, C))
        # arrange dimensions
        inputs = T.transpose(inputs, (2,3,0,1))
        return inputs
        
    def regularize(self) -> T.Tensor:
        """
            Regularization of layer. This layer do not have regularizable parameter.
        """
        return T.Tensor(0.)



class AveragePool3D(Layer):
    """
        AveragePool3D layer implemented by @Author based on AveragePool2D.

        AveragePool is getting average feature of data.

        Arguments for initialization :
        ------------------------------

        kernel              : Size of kernel (Height, Width, Depth). It should declared seperately.
            >>> type        : tuple
            >>> Default     : (2,2,2)

        stride              : Stride of kernel (Height, Width, Depth). It should declared seperately.
            >>> type        : tuple
            >>> Default     : (2,2,2)

        padding             : How padding applied. 'same' and 'valid' is accepted padding types. 'valid' 
            is not apply padding. 'same' is applying paddiing to make output same size w.r.t `Height`, `Width`
            and `Depth`. 
            >>> type        : string
            >>> Default     : 'valid'

        input_shape         : If AveragePool3D is first layer of model, input_shape should be declared.
                            Shape will be in form of (channel, depth, height, width).
            >>> type        : tuple
            >>> Default     : None

        Implementation done by finding max values index and getting them.
    """
    def __init__(self,
                kernel = (2,2,2),
                stride = (2,2,2),
                padding = 'valid',
                input_shape = None,
                **kwargs):
        super(AveragePool3D, self).__init__(**kwargs)
        assert np.array(kernel).size == 3, 'Make sure that kernel size of AveragePool3D has 3 dimension such as (2,2,2), get : ' + str(kernel)
        assert np.array(stride).size == 3, 'Make sure that stride size of AveragePool3D has 3 dimension such as (2,2,2), get : ' + str(stride)
        self._input_shape = input_shape
        self._K_shape = kernel
        self._stride = stride
        self._padding = padding.lower()

    def __call__(self, params) -> None:
        self._thisLayer = params['layer_number']
        params['layer_name'].append('AveragePool3D')
        params['activation'].append('none')
        params['#parameters'].append(0)

        # If AveragePool3D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            # get channel, depth, width and height of data
            self._C, self._D, self._H, self._W = self._input_shape
        else:
            # AveragePool3D layer is not first layer. So get channel, width and height
            # of data from previous layer output shape
            self._C, self._D, self._H, self._W = params['layer_output_shape'][self._thisLayer - 1]

        if self._padding == 'same':
            # To handle even size kernel padding, we put more constant to right and bottom side 
            # of axis like Tensorflow. 
            # Therefore we need to specify axis's each side padding seperately.
            self.H_out =  int(np.ceil(self._H / self._stride[0] ))
            self.W_out =  int(np.ceil(self._W / self._stride[1] ))
            self.D_out =  int(np.ceil(self._D / self._stride[2] ))
            print(self.H_out, self.W_out, self.D_out)
            pd_h = max((self.H_out - 1) * self._stride[0] + self._K_shape[0] - self._H, 0) # height padding 
            pd_w = max((self.W_out - 1) * self._stride[1] + self._K_shape[1] - self._W, 0) # width padding
            pd_d = max((self.D_out - 1) * self._stride[2] + self._K_shape[2] - self._D, 0) # depth padding
            pd_top = int(pd_h / 2) # top side padding 
            pd_bot = int(pd_h - pd_top) # bottom side padding 
            pd_left = int(pd_w / 2) # left side paddding 
            pd_right = int(pd_w - pd_left) # rights side padding 
            pd_front = int(pd_d // 2) # front side padding 
            pd_back = int(pd_d - pd_front) # back side padding 
        else:
            pd_top, pd_bot, pd_left, pd_right, pd_front, pd_back = 0, 0, 0, 0, 0, 0 
            self.H_out =  int((self._H - self._K_shape[0] ) / self._stride[0] + 1)
            self.W_out =  int((self._W - self._K_shape[1] ) / self._stride[1] + 1)
            self.D_out =  int((self._D - self._K_shape[2] ) / self._stride[2] + 1)
        
        
        self._P = ((pd_top, pd_bot), (pd_left, pd_right), (pd_front, pd_back))   

        self._output_shape = (self._C, self.D_out, self.H_out, self.W_out)

        # add output shape to model params without batch_size
        params['layer_output_shape'].append(self._output_shape)
        # add channel number to layer neuron number into model params
        params['model_neuron'].append(self._C)

    def _init_trainable(self, params):
        # AveragePool3D layer has no initialized parameters. Thus, pass it.
        pass

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of AveragePool3D layer.
        '''
        # apply padding
        if self._padding == 'same':
            inputs.value = np.pad(inputs.value, \
                ((0,0),(0,0),(self._P[2][0],self._P[2][1]),(self._P[0][0], self._P[0][1]),(self._P[1][0], self._P[1][1])), \
                    mode='constant', constant_values=0)
        # gettin input shapes separately
        N, C, D, H, W = inputs.shape
        # reshape inputs
        inputs = T.reshape(inputs, (N * C, 1, D, H, W))
        # findin pooling locations
        l, k, i, j = conv_util.get_conv3D_indices(inputs.shape, self._K_shape, self._stride, self._output_shape)
        # getting local spaces
        inputs = inputs[:,l, k,i,j]
        # arrange dimensions
        inputs = T.transpose(inputs,(1,2,0))
        # flat local spaces
        inputs = T.reshape(inputs, (self._K_shape[0] * self._K_shape[1] * self._K_shape[2] , -1)) 
        # find mean values through axis 0
        inputs = T.mean(inputs, axis=0) 
        # reshape it to output
        inputs = T.reshape(inputs, (self.D_out, self.H_out, self.W_out, N, C))
        # arrange dimensions
        inputs = T.transpose(inputs, (3,4,0,1,2))
        return inputs

    def regularize(self) -> T.Tensor:
        """
            Regularization of layer. This layer do not have regularizable parameter.
        """
        return T.Tensor(0.)



class Dropout(Layer):
    """
        Dropout layer implementation which can be found on 
        https://cs231n.github.io/convolutional-networks/#overview.

        Dropout is one of regularization mechanism. It killing/deactivate
         neuron temporary for reduce of possibility of overfitting.

        Arguments for initialization :
        ------------------------------

        p                   : Rate of deactivate neuron.
            >>> type        : float
            >>> Default     : 0.

    """
    def __init__(self,
                p = 0.,
                **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self._drop_prob = p


    def __call__(self, params) -> None:
        self._thisLayer = params['layer_number']
        params['layer_name'].append('Dropout (' + str(self._drop_prob) + ')')
        params['activation'].append('none')
        params['#parameters'].append(0)

        # Dropout layer is not first layer. So output shape from previous layer
        self._output_shape = params['layer_output_shape'][self._thisLayer - 1]

        # add output shape to model params without batch_size
        params['layer_output_shape'].append(self._output_shape)
        # add previous layer neuron number to model params
        params['model_neuron'].append(params['model_neuron'][self._thisLayer - 1])


    def _init_trainable(self, params):
        # Dropout layer has no initialized parameters. Thus, pass it.
        pass

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of Dropout layer.

            Dropout is not used in testing. Therefore, it should be known that this layer
            is calling in testing or training. 
        '''
        if train:
            return T.dropout(inputs, self._drop_prob)
        else:
            return T.dropout(inputs, 0.0)

    def regularize(self) -> T.Tensor:
        """
            Regularization of layer. This layer do not have regularizable parameter.
        """
        return T.Tensor(0.)



class BatchNormalization(Layer):
    """
        Batch Normalization layer implementation. 
        
        Implementation based on https://arxiv.org/abs/1502.03167 and Tensorflow - Keras.

        Batch Normalization can be used after hidden layer of MLP.
        In Convolutional NN, before activation layer or after activation layer, BN can be used. 
        General suggestion is after activation layer.

        Arguments for initialization :
        -----------------------------

        momentum                        : Momentum of running mean and variance. [0,1)
            >>> type                    : float
            >>> Default                 : 0.99
            
        epsilon                         : Constant value to handle division zero error.
            >>> type                    : float
            >>> Default                 : 1e-3 = 0.001
            
        use_gamma                       : Set whether use gamma scaling factor or not.
            >>> type                    : bool
            >>> Default                 : True
                    
        use_beta                        : Set whether use beta offset or not.
            >>> type                    : bool
            >>> Default                 : True
            
        gamma_initializer               : Initializer method of gamma value.
            >>> type                    : str or custom initializer class
            >>> Default                 : 'ones_init'
            
        beta_initializer                : Initializer method of beta value.
            >>> type                    : str or custom initializer class
            >>> Default                 : 'zeros_init'

        running_mean_initializer        : Initializer method of running mean value.
            >>> type                    : str or custom initializer class
            >>> Default                 : 'zeros_init'

        running_var_initializer         : Initializer method of running variance value.
            >>> type                    : str or custom initializer class
            >>> Default                 : 'ones_init'

        gamma_regularizer               : Regularizer method of gamma value.
            >>> type                    : regularizer class
            >>> Default                 : None

        beta_regularizer                : Regularizer method of beta value.
            >>> type                    : regularizer class
            >>> Default                 : None   

    """
    def __init__(self,
                momentum = 0.99,
                epsilon = 1e-3,
                use_gamma = True,
                use_beta = True,
                gamma_initializer = 'ones_init',
                beta_initializer = 'zeros_init',
                running_mean_initializer = 'zeros_init',
                running_var_initializer = 'ones_init',
                gamma_regularizer = None,
                beta_regularizer = None,
                **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self._momentum = momentum
        self._epsilon = epsilon
        self._use_gamma = use_gamma
        self._use_beta = use_beta
        self._gamma_initializer = gamma_initializer
        self._beta_initializer = beta_initializer
        self._running_mean_initializer = running_mean_initializer
        self._running_var_initializer = running_var_initializer
        self._gamma_regularizer = gamma_regularizer
        self._beta_regularizer = beta_regularizer

    def __call__(self, params) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        self._thisLayer = params['layer_number']
        params['layer_name'].append('Batch Normalization')
        params['activation'].append('none')
        params['model_neuron'].append(params['model_neuron'][self._thisLayer - 1])
        params['layer_output_shape'].append(params['layer_output_shape'][self._thisLayer - 1])
        self._init_trainable(params)

    def _init_trainable(self, params):
        '''
            Initialization of batch norm. layer's trainable variables and running mean & variance.
        '''
        # set gamma initializer
        if isinstance(self._gamma_initializer, str):
            _init = self._gamma_initializer.lower()
            self._gamma_init = ID[_init]()
        else:
            self._gamma_init = self._gamma_initializer
        # set beta initializer 
        if isinstance(self._beta_initializer, str):
            _init = self._beta_initializer.lower()
            self._beta_init = ID[_init]()
        else:
            self._beta_init = self._beta_initializer
        # set running mean initializer
        if isinstance(self._running_mean_initializer, str):
            _init = self._running_mean_initializer.lower()
            self._r_mean_init = ID[_init]()
        else:
            self._r_mean_init = self._running_mean_initializer
        # set running var initializer
        if isinstance(self._running_var_initializer, str):
            _init = self._running_var_initializer.lower()
            self._r_var_init = ID[_init]()
        else:
            self._r_var_init = self._running_mean_initializer
        
        pre_layer_output = np.array([params['layer_output_shape'][self._thisLayer - 1]])
        
        # if BN after dense layer, parameter shape is different than after conv layers.
        if len(pre_layer_output.shape) == 1:
            self._gamma = self._gamma_init.get_init(shape=pre_layer_output)
            self._beta = self._beta_init.get_init(shape=pre_layer_output)
            self._r_mean = self._r_mean_init.get_init(shape=pre_layer_output)
            self._r_var = self._r_var_init.get_init(shape=pre_layer_output)
            params['#parameters'].append(pre_layer_output[0] * 4)
        else:
            self._trainable_shape = [1,int(pre_layer_output[0][0])]
            for _ in range(len(pre_layer_output[0])-1):
                self._trainable_shape.append(1)
            self._axis = list(np.arange(len(pre_layer_output.shape)+2))
            del self._axis[1]
            self._gamma = self._gamma_init.get_init(shape=self._trainable_shape)
            self._beta = self._beta_init.get_init(shape=self._trainable_shape)
            self._r_mean = self._r_mean_init.get_init(shape=self._trainable_shape)
            self._r_var = self._r_var_init.get_init(shape=self._trainable_shape)
            params['#parameters'].append(pre_layer_output[0][0] * 4)

        if self._use_gamma:
            self._trainable.append(T.Tensor(self._gamma.astype(np.float32), have_grad=True))
        if self._use_beta:
            self._trainable.append(T.Tensor(self._beta.astype(np.float32), have_grad=True))


        
    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of dense layer.
        '''
        if train:
            # calculate mean and variance of inputs
            if len(inputs.shape) == 2:
                mu = np.mean(inputs.value, axis=0, dtype=np.float32)
                var = np.mean((inputs.value - mu)**2, axis=0, dtype=np.float32)
            else:         
                mu = np.mean(inputs.value, axis=tuple(self._axis), keepdims=True, dtype=np.float32)
                var = np.var(inputs.value, axis=tuple(self._axis), keepdims=True, dtype=np.float32)

            #print(var.shape, self._trainable[0].shape)
            # calculate division part of formula
            div = 1./ np.sqrt(var + self._epsilon)

            # if scale factor is used, scale it
            if self._use_gamma:
                div =  div * self._trainable[0]

            # formula apply
            #print(inputs.shape, div.shape, mu.shape)
            inputs = inputs * div - mu * div

            # if offset is used, offset it
            if self._use_beta:
                inputs = inputs + self._trainable[1]
        
            # calculate running values
            self._r_mean = self._r_mean * self._momentum + (1.0 - self._momentum) * mu
            self._r_var = self._r_var * self._momentum + (1.0 - self._momentum) * var

        else:
            # calculate division part w.r.t running values
            div = 1. / np.sqrt(self._r_var + self._epsilon)

            # if scale factor is used, scale it
            if self._use_gamma:
                div = div * self._trainable[0]

            # formula apply
            inputs = inputs * div - self._r_mean * div

            # if offset is used, offset it
            if self._use_beta:
                inputs = inputs + self._trainable[1]
        
        return inputs

    def regularize(self) -> T.Tensor:
        """
            Regularization of layer. 
        """
        _res = T.Tensor(0.)
        if self._gamma_regularizer:
            _res += self._gamma_regularizer.compute(self._trainable[0])
        if self._beta_regularizer:
            _res += self._beta_regularizer.compute(self._trainable[1])
        return _res



class SimpleRNN(Layer):
    """
        SimpleRNN layer implementation based on Keras' implementation done by @Author.

        Arguments for initialization :
        ------------------------------

        cell                : Number of cell.
            >>> type        : int
            >>> Default     : 1

        activation_function : Activation function of SimpleRNN method.
            >>> type        : string
            >>> Default     : 'tanh'

        initializer         : Initialize method of input kernel.
            >>> type        : str or custom initializer class
            >>> Default     : 'xavier_uniform'

        hidden_initializer  : Initialize method of hidden state kernel. 
            >>> type        : string
            >>> Default     : 'orthogonal'

        bias_initializer    : Layer's bias's initialization method.
            >>> type        : str or custom initializer class
            >>> Default     : 'zeros_init'

        return_sequences    : Returning sequencial of output. 
            >>> type        : bool
            >>> Default     : False

        return_state        : Returning last state of layer. It used with `return_sequences=True`. 
            It returns sequential output of layer and last state respectly.
            >>> type        : bool
            >>> Default     : False

        kernel_regularizer  : Regularizer method of kernels of layer.
            >>> type        : regularizer class
            >>> Default     : None

        hidden_regularizer  : Regularizer method of hidden state of layer.
            >>> type        : regularizer class
            >>> Default     : None

        bias_regularizer    : Regularizer method of biases of layer.
            >>> type        : regularizer class
            >>> Default     : None

        use_bias            : Bool of using bias during calculation.
            >>> type        : bool
            >>> Default     : True

        input_shape         : If SimpleRNN is first layer of model, input_shape should be declared.
                            Shape will be in form of (sequential length, data width (word_size)).
            >>> type        : tuple
            >>> Default     : None

        Arguments for compute method is tensor of previous method in proper size.
    """
    def __init__(self,
                cell = 1,
                activation_function = 'tanh',
                initializer = 'xavier_uniform',
                hidden_initializer = 'orthogonal',
                bias_initializer = 'zeros_init',
                return_sequences = False,
                return_state = False,
                kernel_regularizer = None,
                hidden_regularizer = None,
                bias_regularizer = None,
                use_bias = True,
                input_shape = None,
                **kwargs):
        super(SimpleRNN, self).__init__(**kwargs)
        self._cell = cell
        self._input_shape = input_shape
        self._activation = activation_function
        self._ret_seq = return_sequences
        self._ret_sta = return_state
        self._bias = use_bias
        self._initialize_method = initializer
        self._hidden_initializer = hidden_initializer
        self._bias_initialize_method = bias_initializer
        self._set_initializer()
        self._kernel_regularizer = kernel_regularizer
        self._hidden_regularizer = hidden_regularizer
        self._bias_regularizer = bias_regularizer

    def __call__(self, params) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        self._thisLayer = params['layer_number']
        params['layer_name'].append('SimpleRNN : ' + str(self._activation))
        params['activation'].append(str(self._activation))

        # If SimpleRNN layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            assert len(self._input_shape) == 2, 'Make sure that input of SimpleRNN has 2 dimension without batch such as (10,5).'
            # get sequencial lenght and input size
            self._seq_len, self._inp_size = self._input_shape
        else:
            # SimpleRNN layer is not first layer. So get sequential lenght and input size 
            # ofrom previous layer output shape
            assert self._thisLayer != 0, 'First layer of SimpleRNN should have input_shape!'
            if isinstance(params['layer_output_shape'][self._thisLayer - 1], int):
                params['layer_output_shape'][self._thisLayer - 1] = [params['layer_output_shape'][self._thisLayer - 1]]
            assert len(params['layer_output_shape'][self._thisLayer - 1]) == 2, 'Previous RNN layer`s `return_sequences` should be True'
            self._seq_len, self._inp_size = params['layer_output_shape'][self._thisLayer - 1]  

        if self._ret_seq:
            self._output_shape = (self._seq_len, self._cell)
        else:
            self._output_shape = (self._cell)


        # add output shape to model params without batch_size
        params['layer_output_shape'].append(self._output_shape)
        # add layer neuron number to model params
        params['model_neuron'].append(self._cell)
        # add number of parameters 
        params['#parameters'].append(self._cell * ( self._cell + self._inp_size + 1))

        # if activation function is `str` create caller.
        if isinstance(self._activation, str):
            self._actCaller = self._actFuncCaller[self._activation]()
        else:
            self._actCaller = self._activation

        self._init_trainable(params)

    def _init_trainable(self, params):
        '''
            Initialization of SimpleRNN layer's trainable variables.
        '''
        # create kernels and biases
        if isinstance(self._hidden_initializer, str):
            _init = self._hidden_initializer.lower()
            self._hidden_init = ID[_init]()
        else:
            self._hidden_init = self._hidden_initializer

        _k_hh = self._hidden_init.get_init(shape=(self._cell, self._cell)) 
        _k_hx = self._initializer.get_init(shape=(self._inp_size, self._cell)) 
        _b_h = self._bias_initializer.get_init(shape=(1,self._cell)) 

        # make kernels as  tensor and add them to trainable list
        self._trainable.append(T.Tensor(_k_hh.astype(np.float32), have_grad=True))
        self._trainable.append(T.Tensor(_k_hx.astype(np.float32), have_grad=True))
        self._trainable.append(T.Tensor(_b_h.astype(np.float32), have_grad=True))

        
    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of SimpleRNN layer.
        '''
        # initializer hidden matrix as zeros
        h_init = ID['zeros_init']()
        h = T.Tensor(h_init.get_init((inputs.shape[0],self._cell)))

        # sequential output holder
        return_seq = T.Tensor(np.empty((inputs.shape[0],self._cell)))
        
        # for each sequencial data 
        for s in range(self._seq_len):
            # finding value to activate
            tmp = T.matmul(inputs[:,s,:], self._trainable[1]) + T.matmul(h, self._trainable[0])
            # adding if use_bias is true 
            if self._bias:
                tmp += self._trainable[2]
            # activate value
            h = self._actCaller.activate(tmp)
            # add sequential output
            if self._ret_seq:
                if s == 0:
                    return_seq = h
                else:    
                    return_seq = T.append(return_seq, h, 0)            
                    
        if self._ret_seq:
            return_seq = T.reshape(return_seq, (self._seq_len,-1, self._cell))
            if self._ret_sta:
                return T.transpose(return_seq, (1,0,2)), h 
            return T.transpose(return_seq, (1,0,2))
        else:
            return h



    def regularize(self) -> T.Tensor:
        """
            Regularization of layer.
        """
        _res = T.Tensor(0.)
        if self._kernel_regularizer:
            _res += self._kernel_regularizer.compute(self._trainable[1])
        if self._hidden_regularizer:
            _res += self._hidden_regularizer.compute(self._trainable[0])
        if self._bias_regularizer:
            _res += self._bias_regularizer.compute(self._trainable[2])
        return _res



class LSTM(Layer):
    """
        LSTM layer implementation based on Keras' implementation done by @Author.

        Arguments for initialization :
        ------------------------------

        cell                        : Number of cell.
            >>> type                : int
            >>> Default             : 1

        activation_function         : Activation function of kernels of LSTM layer.
            >>> type                : string
            >>> Default             : 'tanh'

        hidden_activation_function  : Activation function of hidden state kernels of LSTM layer.
            >>> type                : string
            >>> Default             : 'tanh'

        initializer                 : Initialize method of input kernel.
            >>> type                : str or custom initializer class
            >>> Default             : 'xavier_uniform'

        hidden_initializer          : Initialize method of hidden state kernel. 
            >>> type                : string
            >>> Default             : 'orthogonal'

        bias_initializer            : Layer's bias's initialization method.
            >>> type                : str or custom initializer class
            >>> Default             : 'zeros_init'

        return_sequences            : Returning sequencial of output. 
            >>> type                : bool
            >>> Default             : False

        return_state                : Returning last state of layer. It used with `return_sequences=True`. 
            It returns sequential output of layer and last state respectly.
            >>> type                : bool
            >>> Default             : False

        kernel_regularizer          : Regularizer method of kernels of layer.
            >>> type                : regularizer class
            >>> Default             : None

        hidden_regularizer          : Regularizer method of hidden state of layer.
            >>> type                : regularizer class
            >>> Default             : None

        bias_regularizer            : Regularizer method of biases of layer.
            >>> type                : regularizer class
            >>> Default             : None

        use_bias                    : Bool of using bias during calculation.
            >>> type                : bool
            >>> Default             : True

        use_forget_bias             : Bool of using forget bias which initialize as ones.
            >>> type                : bool
            >>> Default             : True

        input_shape                 : If LSTM is first layer of model, input_shape should be declared.
                                    Shape will be in form of (sequential length, data width (word_size)).
            >>> type                : tuple
            >>> Default             : None

        Arguments for compute method is tensor of previous method in proper size.
    """
    def __init__(self,
                cell = 1,
                activation_function = 'tanh',
                hidden_activation_function = 'sigmoid',
                initializer = 'xavier_uniform',
                hidden_initializer = 'orthogonal',
                bias_initializer = 'zeros_init',
                return_sequences = False,
                return_state = False,
                kernel_regularizer = None,
                hidden_regularizer = None,
                bias_regularizer = None,
                use_bias = True,
                use_forget_bias = True,
                input_shape = None,
                **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self._cell = cell
        self._input_shape = input_shape
        self._activation = activation_function
        self._hidden_activation = hidden_activation_function
        self._ret_seq = return_sequences
        self._ret_sta = return_state
        self._bias = use_bias
        self._forget_bias = use_forget_bias
        self._initialize_method = initializer
        self._hidden_initializer = hidden_initializer
        self._bias_initialize_method = bias_initializer
        self._set_initializer()
        self._kernel_regularizer = kernel_regularizer
        self._hidden_regularizer = hidden_regularizer
        self._bias_regularizer = bias_regularizer

    def __call__(self, params) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        self._thisLayer = params['layer_number']
        act_name = str('(Kernel : ' + str(self._activation)+ ' & Hidden : ' + str(self._hidden_activation) + ')')
        params['layer_name'].append('LSTM : ' + act_name)
        params['activation'].append(act_name)

        # If LSTM layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            assert len(self._input_shape) == 2, 'Make sure that input of LSTM has 2 dimension without batch such as (10,5).'
            # get sequencial lenght and input size
            self._seq_len, self._inp_size = self._input_shape
        else:
            # SimpleRNN layer is not first layer. So get sequential lenght and input size 
            # ofrom previous layer output shape
            assert self._thisLayer != 0, 'First layer of LSTM should have input_shape!'
            if isinstance(params['layer_output_shape'][self._thisLayer - 1], int):
                params['layer_output_shape'][self._thisLayer - 1] = [params['layer_output_shape'][self._thisLayer - 1]]
            assert len(params['layer_output_shape'][self._thisLayer - 1]) == 2, 'Previous RNN layer`s `return_sequences` should be True'
            self._seq_len, self._inp_size = params['layer_output_shape'][self._thisLayer - 1]  

        if self._ret_seq:
            self._output_shape = (self._seq_len, self._cell)
        else:
            self._output_shape = (self._cell)


        # add output shape to model params without batch_size
        params['layer_output_shape'].append(self._output_shape)
        # add layer neuron number to model params
        params['model_neuron'].append(self._cell)
        # add number of parameters 
        params['#parameters'].append(4 * self._cell * ( self._cell + self._inp_size + 1))

        # if activation function is `str` create caller.
        if isinstance(self._activation, str):
            self._actCaller = self._actFuncCaller[self._activation]()
        else:
            self._actCaller = self._activation

        # if hidden activation function is `str` create caller.
        if isinstance(self._hidden_activation, str):
            self._hidden_actCaller = self._actFuncCaller[self._activation]()
        else:
            self._hidden_actCaller = self._activation

        self._init_trainable(params)

    def _init_trainable(self, params):
        '''
            Initialization of LSTM layer's trainable variables.
        '''
        # create kernels and biases
        if isinstance(self._hidden_initializer, str):
            _init = self._hidden_initializer.lower()
            self._hidden_init = ID[_init]()
        else:
            self._hidden_init = self._hidden_initializer

        _k_hh = self._hidden_init.get_init(shape=(self._cell, self._cell * 4))
        _k_hh_f = _k_hh[:,:self._cell]
        _k_hh_i = _k_hh[:,self._cell:self._cell*2]
        _k_hh_c = _k_hh[:,self._cell*2:self._cell*3]
        _k_hh_o = _k_hh[:,self._cell*3:self._cell*4]        
        
        _k_hx = self._initializer.get_init(shape=(self._inp_size, self._cell * 4)) 
        _k_hx_f = _k_hx[:, :self._cell]
        _k_hx_i = _k_hx[:, self._cell:self._cell*2]
        _k_hx_c = _k_hx[:, self._cell*2:self._cell*3]
        _k_hx_o = _k_hx[:, self._cell*3:self._cell*4]
 
        _b = self._bias_initializer.get_init(shape=(1,self._cell * 4)) 
        if self._forget_bias:
            _b_h_f = ID['ones_init']().get_init(shape=(1, self._cell))
        else:
            _b_h_f = _b[:, :self._cell]
        _b_h_i = _b[:, self._cell:self._cell*2]
        _b_h_c = _b[:, self._cell*2:self._cell*3]
        _b_h_o = _b[:, self._cell*3:self._cell*4]

        # make kernels as tensor and add them to trainable list        
        self._trainable.append(T.Tensor(_k_hh_f.astype(np.float32), have_grad=True)) # 0
        self._trainable.append(T.Tensor(_k_hh_i.astype(np.float32), have_grad=True)) # 1
        self._trainable.append(T.Tensor(_k_hh_c.astype(np.float32), have_grad=True)) # 2
        self._trainable.append(T.Tensor(_k_hh_o.astype(np.float32), have_grad=True)) # 3
        
        self._trainable.append(T.Tensor(_k_hx_f.astype(np.float32), have_grad=True)) # 4 
        self._trainable.append(T.Tensor(_k_hx_i.astype(np.float32), have_grad=True)) # 5
        self._trainable.append(T.Tensor(_k_hx_c.astype(np.float32), have_grad=True)) # 6
        self._trainable.append(T.Tensor(_k_hx_o.astype(np.float32), have_grad=True)) # 7

        self._trainable.append(T.Tensor(_b_h_f.astype(np.float32), have_grad=True)) # 8
        self._trainable.append(T.Tensor(_b_h_i.astype(np.float32), have_grad=True)) # 9
        self._trainable.append(T.Tensor(_b_h_c.astype(np.float32), have_grad=True)) # 10
        self._trainable.append(T.Tensor(_b_h_o.astype(np.float32), have_grad=True)) # 11

        
    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of LSTM layer.
        '''
        # initializer hidden matrix and cell state as zeros
        h_init = ID['zeros_init']()
        h = T.Tensor(h_init.get_init((inputs.shape[0],self._cell)))
        cell_state = T.Tensor(h_init.get_init((inputs.shape[0],self._cell)))

        # sequential output holder        
        return_seq = T.Tensor(np.empty((inputs.shape[0],self._cell)))

        # for each sequencial data 
        for s in range(self._seq_len):
            # finding value to activate
            f = T.matmul(inputs[:,s,:], self._trainable[4]) + T.matmul(h.value , self._trainable[0])
            i = T.matmul(inputs[:,s,:], self._trainable[5]) + T.matmul(h.value , self._trainable[1])
            c = T.matmul(inputs[:,s,:], self._trainable[6]) + T.matmul(h.value , self._trainable[2])
            o = T.matmul(inputs[:,s,:], self._trainable[7]) + T.matmul(h.value , self._trainable[3])
            # adding if use_bias is true 
            if self._bias:
                f += self._trainable[8] 
                i += self._trainable[9]
                c += self._trainable[10]
                o += self._trainable[11]
            # activate value
            ft = self._hidden_actCaller.activate(f)
            it = self._hidden_actCaller.activate(i)
            ct = self._actCaller.activate(c)
            ot = self._hidden_actCaller.activate(o)
            # calculate cell state
            cell_state = ft * cell_state + it * ct
            # calculate hidden state output
            h = ot * self._actCaller.activate(cell_state)
        
            # add sequential output
            if self._ret_seq:
                if s == 0:
                    return_seq = h
                else:    
                    return_seq = T.append(return_seq, h, 0)                    

        if self._ret_seq:
            return_seq = T.reshape(return_seq, (self._seq_len,-1, self._cell))
            if self._ret_sta:
                return T.transpose(return_seq, (1,0,2)), h 
            return T.transpose(return_seq, (1,0,2))
        else:
            return h

    def regularize(self) -> T.Tensor:
        """
            Regularization of layer.
        """
        _res = T.Tensor(0.)
        if self._kernel_regularizer:
            _res += self._kernel_regularizer.compute(self._trainable[4])
            _res += self._kernel_regularizer.compute(self._trainable[5])
            _res += self._kernel_regularizer.compute(self._trainable[6])
            _res += self._kernel_regularizer.compute(self._trainable[7])
        if self._hidden_regularizer:
            _res += self._hidden_regularizer.compute(self._trainable[0])
            _res += self._hidden_regularizer.compute(self._trainable[1])
            _res += self._hidden_regularizer.compute(self._trainable[2])
            _res += self._hidden_regularizer.compute(self._trainable[3])
        if self._bias_regularizer:
            _res += self._bias_regularizer.compute(self._trainable[8])
            _res += self._bias_regularizer.compute(self._trainable[9])
            _res += self._bias_regularizer.compute(self._trainable[10])
            _res += self._bias_regularizer.compute(self._trainable[11])
        return _res



class GRU(Layer):
    """
        GRU layer implementation based on Keras' implementation done by @Author.

        Arguments for initialization :
        ------------------------------

        cell                        : Number of cell.
            >>> type                : int
            >>> Default             : 1

        activation_function         : Activation function of kernels of GRU layer.
            >>> type                : string
            >>> Default             : 'tanh'

        hidden_activation_function  : Activation function of hidden state kernels of GRU layer.
            >>> type                : string
            >>> Default             : 'tanh'

        initializer                 : Initialize method of input kernel.
            >>> type                : str or custom initializer class
            >>> Default             : 'xavier_uniform'

        hidden_initializer          : Initialize method of hidden state kernel. 
            >>> type                : string
            >>> Default             : 'orthogonal'

        bias_initializer            : Layer's bias's initialization method.
            >>> type                : str or custom initializer class
            >>> Default             : 'zeros_init'

        return_sequences            : Returning sequencial of output. 
            >>> type                : bool
            >>> Default             : False

        return_state                : Returning last state of layer. It used with `return_sequences=True`. 
            It returns sequential output of layer and last state respectly.
            >>> type                : bool
            >>> Default             : False

        kernel_regularizer          : Regularizer method of kernels of layer.
            >>> type                : regularizer class
            >>> Default             : None

        hidden_regularizer          : Regularizer method of hidden state of layer.
            >>> type                : regularizer class
            >>> Default             : None

        bias_regularizer            : Regularizer method of biases of layer.
            >>> type                : regularizer class
            >>> Default             : None

        use_bias                    : Bool of using bias during calculation.
            >>> type                : bool
            >>> Default             : True

        input_shape                 : If LSTM is first layer of model, input_shape should be declared.
                                    Shape will be in form of (sequential length, data width (word_size)).
            >>> type                : tuple
            >>> Default             : None

        Arguments for compute method is tensor of previous method in proper size.
    """
    def __init__(self,
                cell = 1,
                activation_function = 'tanh',
                hidden_activation_function = 'sigmoid',
                initializer = 'xavier_uniform',
                hidden_initializer = 'orthogonal',
                bias_initializer = 'zeros_init',
                return_sequences = False,
                return_state = False,
                kernel_regularizer = None,
                hidden_regularizer = None,
                bias_regularizer = None,
                use_bias = True,
                input_shape = None,
                **kwargs):
        super(GRU, self).__init__(**kwargs)
        self._cell = cell
        self._input_shape = input_shape
        self._activation = activation_function
        self._hidden_activation = hidden_activation_function
        self._ret_seq = return_sequences
        self._ret_sta = return_state
        self._bias = use_bias
        self._initialize_method = initializer
        self._hidden_initializer = hidden_initializer
        self._bias_initialize_method = bias_initializer
        self._set_initializer()
        self._kernel_regularizer = kernel_regularizer
        self._hidden_regularizer = hidden_regularizer
        self._bias_regularizer = bias_regularizer

    def __call__(self, params) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        self._thisLayer = params['layer_number']
        act_name = str('(Kernel : ' + str(self._activation)+ ' & Hidden : ' + str(self._hidden_activation) + ')')
        params['layer_name'].append('GRU : ' + act_name )
        params['activation'].append(act_name)

        # If GRU layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            assert len(self._input_shape) == 2, 'Make sure that input of GRU has 2 dimension without batch such as (10,5).'
            # get sequencial lenght and input size
            self._seq_len, self._inp_size = self._input_shape
        else:
            # SimpleRNN layer is not first layer. So get sequential lenght and input size 
            # ofrom previous layer output shape
            assert self._thisLayer != 0, 'First layer of GRU should have input_shape!'
            if isinstance(params['layer_output_shape'][self._thisLayer - 1], int):
                params['layer_output_shape'][self._thisLayer - 1] = [params['layer_output_shape'][self._thisLayer - 1]]
            assert len(params['layer_output_shape'][self._thisLayer - 1]) == 2, 'Previous RNN layer`s `return_sequences` should be True'
            self._seq_len, self._inp_size = params['layer_output_shape'][self._thisLayer - 1]  

        if self._ret_seq:
            self._output_shape = (self._seq_len, self._cell)
        else:
            self._output_shape = (self._cell)

        # add output shape to model params without batch_size
        params['layer_output_shape'].append(self._output_shape)
        # add layer neuron number to model params
        params['model_neuron'].append(self._cell)
        # add number of parameters 
        params['#parameters'].append(3 * self._cell * ( self._cell + self._inp_size + 1))

        # if activation function is `str` create caller.
        if isinstance(self._activation, str):
            self._actCaller = self._actFuncCaller[self._activation]()
        else:
            self._actCaller = self._activation

        # if hidden activation function is `str` create caller.
        if isinstance(self._hidden_activation, str):
            self._hidden_actCaller = self._actFuncCaller[self._activation]()
        else:
            self._hidden_actCaller = self._activation

        self._init_trainable(params)

    def _init_trainable(self, params):
        '''
            Initialization of GRU layer's trainable variables.
        '''
        # create kernels and biases
        if isinstance(self._hidden_initializer, str):
            _init = self._hidden_initializer.lower()
            self._hidden_init = ID[_init]()
        else:
            self._hidden_init = self._hidden_initializer

        _k_hh = self._hidden_init.get_init(shape=(self._cell, self._cell * 3))
        _k_hh_z = _k_hh[:,:self._cell]
        _k_hh_r = _k_hh[:,self._cell:self._cell*2]
        _k_hh_h = _k_hh[:,self._cell*2:self._cell*3]
        
        _k_hx = self._initializer.get_init(shape=(self._inp_size, self._cell * 3)) 
        _k_hx_z = _k_hx[:, :self._cell]
        _k_hx_r = _k_hx[:, self._cell:self._cell*2]
        _k_hx_h = _k_hx[:, self._cell*2:self._cell*3]
 
        _b = self._bias_initializer.get_init(shape=(1,self._cell * 3)) 
        _b_h_z = _b[:, :self._cell]
        _b_h_r = _b[:, self._cell:self._cell*2]
        _b_h_h = _b[:, self._cell*2:self._cell*3]

        # make kernels as  tensor and add them to trainable list        
        self._trainable.append(T.Tensor(_k_hh_z.astype(np.float32), have_grad=True)) # 0
        self._trainable.append(T.Tensor(_k_hh_r.astype(np.float32), have_grad=True)) # 1
        self._trainable.append(T.Tensor(_k_hh_h.astype(np.float32), have_grad=True)) # 2
        
        self._trainable.append(T.Tensor(_k_hx_z.astype(np.float32), have_grad=True)) # 3 
        self._trainable.append(T.Tensor(_k_hx_r.astype(np.float32), have_grad=True)) # 4
        self._trainable.append(T.Tensor(_k_hx_h.astype(np.float32), have_grad=True)) # 5

        self._trainable.append(T.Tensor(_b_h_z.astype(np.float32), have_grad=True)) # 6
        self._trainable.append(T.Tensor(_b_h_r.astype(np.float32), have_grad=True)) # 7
        self._trainable.append(T.Tensor(_b_h_h.astype(np.float32), have_grad=True)) # 8

        
    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of GRU layer.
        '''
        # initializer hidden matrix and cell state as zeros
        h_init = ID['zeros_init']()
        h = T.Tensor(h_init.get_init((inputs.shape[0],self._cell)))
        
        # sequential output holder        
        return_seq = T.Tensor(np.empty((inputs.shape[0],self._cell)))

        # for each sequencial data 
        for s in range(self._seq_len):
            # finding value to activate
            z = T.matmul(inputs[:,s,:], self._trainable[3])  + T.matmul(h.value, self._trainable[0])
            r = T.matmul(inputs[:,s,:], self._trainable[4])  + T.matmul(h.value, self._trainable[1])
            # adding if use_bias is true 
            if self._bias:
                z += self._trainable[6]
                r += self._trainable[7]
            
            # activate value
            z = self._hidden_actCaller.activate(z)
            r = self._hidden_actCaller.activate(r)
            ht = T.matmul(inputs[:,s,:], self._trainable[5]) + T.matmul(r * h.value, self._trainable[2])
            if self._bias:
                ht += self._trainable[8]
            ht = self._actCaller.activate(ht)
            
            # calculate hidden output
            h = z * h + (1-z) * ht

            # add sequential output
            if self._ret_seq:
                if s == 0:
                    return_seq = h
                else:    
                    return_seq = T.append(return_seq, h, 0)                    
        
        if self._ret_seq:
            return_seq = T.reshape(return_seq, (self._seq_len,-1, self._cell))
            if self._ret_sta:
                return T.transpose(return_seq, (1,0,2)), h 
            return T.transpose(return_seq, (1,0,2))
        else:
            return h

    def regularize(self) -> T.Tensor:
        """
            Regularization of layer.
        """
        _res = T.Tensor(0.)
        if self._kernel_regularizer:
            _res += self._kernel_regularizer.compute(self._trainable[3])
            _res += self._kernel_regularizer.compute(self._trainable[4])
            _res += self._kernel_regularizer.compute(self._trainable[5])
        if self._hidden_regularizer:
            _res += self._hidden_regularizer.compute(self._trainable[0])
            _res += self._hidden_regularizer.compute(self._trainable[1])
            _res += self._hidden_regularizer.compute(self._trainable[2])
        if self._bias_regularizer:
            _res += self._bias_regularizer.compute(self._trainable[6])
            _res += self._bias_regularizer.compute(self._trainable[7])
            _res += self._bias_regularizer.compute(self._trainable[8])
        return _res




