"""
    Laywer module of gNet.

    Containing layers : \n
        - Dense
        - Flatten
        - Activation
        - Conv2D
        - MaxPool2D
        - AveragePool2D
        - Dropout
        - Batch Normalization

    Layer should be used with Model Class's add method. Rest of the calculation should be done 
    by NN structure.

    Author : @MGokcayK 
    Create : 04 / 04 / 2020
    Update : 08 / 07 / 2020
                Add some assert for end-user.
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
        params['layer_name'].append('Dense : ' + self._activation)
        params['activation'].append(self._activation)
        params['model_neuron'].append(self._neuronNumber)
        params['layer_output_shape'].append(self._neuronNumber)
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
        self._trainable.append(T.Tensor(_w, have_grad=True))
        self._trainable.append(T.Tensor(_b, have_grad=True))

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of dense layer.
        '''
        _z_layer = inputs @ self._trainable[0] + self._trainable[1]
        return self._actFuncCaller[self._activation].activate(_z_layer)

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

        padding             : How many padding of input. Integer should give declare number of 
                            padding constant. 
            >>> type        : int                            
            >>> Default     : 0

        initialize_method   : Layer initialization method.
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
            >>> type        : bool
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
                padding = 0,
                initialize_method = 'xavier_uniform',
                bias_initializer = 'zeros_init',
                kernel_regularizer = None,
                bias_regularizer = None,
                use_bias = True,
                input_shape = None,
                **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self._input_shape = input_shape
        self._filter = filter
        self._HH = kernel[0]
        self._WW = kernel[1]
        self._stride_row = stride[0]
        self._stride_col = stride[1]
        self._padding = padding
        self._bias = use_bias
        self._initialize_method = initialize_method
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

        # assert (self._H - self._HH + 2 * self._padding) % self._stride_row == 0, \
        #     'Kernel, Stride or Padding property for Height of Input is not proper. Please check them.'
        # assert (self._W - self._WW + 2 * self._padding) % self._stride_col == 0, \
        #     'Kernel, Stride or Padding property for Width of Input is not proper. Please check them.'

        self.H_out =  int((self._H - self._HH + 2 * self._padding) / self._stride_row + 1)
        self.W_out =  int((self._W - self._WW + 2 * self._padding) / self._stride_col + 1)

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
        _K = T.Tensor(_K, have_grad=True)
        # make biases as tensor
        _b = T.Tensor(_b, have_grad=True)
        # add them to trainable list
        self._trainable.append(_K)
        self._trainable.append(_b)

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of Conv2D layer.
        '''
        # getting input shapes separately
        N, C, H, W = inputs.shape
        # base_shape 
        base_shape = inputs.shape
        # padding 
        if self._padding != 0:
            inputs.value = np.pad(inputs.value, ((0, 0), (0, 0), \
                (self._padding, self._padding), (self._padding, self._padding)), mode='constant')
        # location of local space
        k, i, j = conv_util.get_im2col_indices(base_shape, (self._HH, self._WW), (self._stride_row, self._stride_col), self._padding)
        # get local spaces of inputs
        value = inputs[:, k, i, j].value
        # flat the local spaces
        value = value.transpose(1,2,0).reshape((self._HH * self._WW * C, -1))
        # dot product of kernels and local spaces
        inputs = T.dot(T.reshape(self._trainable[0], shape=(self._filter, -1)), T.Tensor(value)  )
        # adding if use_bias is true
        if self._bias:
            inputs.value += self._trainable[1].value
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
        params['layer_name'].append('Activation Layer :' + self._activation)
        params['activation'].append(self._activation)
        params['#parameters'].append(0)
        params['model_neuron'].append(params['model_neuron'][self._thisLayer - 1])
        params['layer_output_shape'].append(params['layer_output_shape'][self._thisLayer - 1])

    def _init_trainable(self, params):
        # Activation layer has no initialized parameters. Thus, pass it.
        pass

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of activation layer.
        '''
        return self._actFuncCaller[self._activation].activate(inputs)

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

        input_shape         : If MaxPool2D is first layer of model, input_shape shoudl be declared.
            >>> type        : tuple
            >>> Default     : None

        Implementation done by finding max values index and getting them.
    """
    def __init__(self,
                kernel = (2,2),
                stride = (2,2),
                input_shape = None,
                **kwargs):
        super(MaxPool2D, self).__init__(**kwargs)
        self._input_shape = input_shape
        self._HH = kernel[0]
        self._WW = kernel[1]
        self._stride_row = stride[0]
        self._stride_col = stride[1]

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

        # assert (self._H - self._HH ) % self._stride_row == 0, \
        #     'MaxPool2D height of kernel is not proper. Please check them.'
        # assert (self._W - self._WW ) % self._stride_col == 0, \
        #     'MaxPool2D width of kernel is not proper. Please check them.'

        self.H_out =  int((self._H - self._HH ) / self._stride_row + 1)
        self.W_out =  int((self._W - self._WW ) / self._stride_col + 1)

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
        # gettin input shapes separately
        N, C, H, W = inputs.shape
        # reshape inputs
        inputs = T.reshape(inputs, (N * C, 1, H, W))
        # getting temp input shape
        tmp = inputs.shape
        # findin pooling locations
        k, i, j = conv_util.get_im2col_indices(inputs.shape, (self._HH, self._WW), (self._stride_row, self._stride_col), 0)
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

        input_shape         : If AveragePool2D is first layer of model, input_shape shoudl be declared.
            >>> type            : tuple
            >>> Default         : None

        Implementation done by finding average values.
    """
    def __init__(self,
                kernel = (2,2),
                stride = (2,2),
                input_shape = None,
                **kwargs):
        super(AveragePool2D, self).__init__(**kwargs)
        self._input_shape = input_shape
        self._HH = kernel[0]
        self._WW = kernel[1]
        self._stride_row = stride[0]
        self._stride_col = stride[1]

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

        self.H_out =  int((self._H - self._HH ) / self._stride_row + 1)
        self.W_out =  int((self._W - self._WW ) / self._stride_col + 1)

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
        # gettin input shapes separately
        N, C, H, W = inputs.shape
        # reshape inputs
        inputs = T.reshape(inputs, (N * C, 1, H, W))
        # getting temp input shape
        tmp = inputs.shape
        # findin pooling locations
        k, i, j = conv_util.get_im2col_indices(inputs.shape, (self._HH, self._WW), (self._stride_row, self._stride_col), 0)
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
        
        # if BN after dense layer, parameter shape is different than after conv2D layer.
        if len(pre_layer_output.shape) == 1:
            self._gamma = self._gamma_init.get_init(shape=pre_layer_output)
            self._beta = self._beta_init.get_init(shape=pre_layer_output)
            self._r_mean = self._r_mean_init.get_init(shape=pre_layer_output)
            self._r_var = self._r_var_init.get_init(shape=pre_layer_output)
        else:
            self._gamma = self._gamma_init.get_init(shape=pre_layer_output[0][1])
            self._beta = self._beta_init.get_init(shape=pre_layer_output[0][1])
            self._r_mean = self._r_mean_init.get_init(shape=pre_layer_output[0][1])
            self._r_var = self._r_var_init.get_init(shape=pre_layer_output[0][1])

        p_coef = 2
        if self._use_gamma:
            self._trainable.append(T.Tensor(self._gamma, have_grad=True))
            p_coef += 1
        if self._use_beta:
            self._trainable.append(T.Tensor(self._beta, have_grad=True))
            p_coef += 1

        params['#parameters'].append(len(self._r_var)*p_coef)

        
    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of dense layer.
        '''
        if train:
            # calculate mean and variance of inputs
            if len(inputs.shape) == 2:
                mu = np.mean(inputs.value, axis=0)
                var = np.mean((inputs.value - mu)**2, axis=0)
            else:                
                mu = np.mean(inputs.value, axis=(0,2,3), keepdims=True)
                var = np.var(inputs.value, axis=(0,2,3), keepdims=True)

            # calculate division part of formula
            div = 1./ np.sqrt(var + self._epsilon)

            # if scale factor is used, scale it
            if self._use_gamma:
                div =  div * self._trainable[0]

            # formula apply
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

