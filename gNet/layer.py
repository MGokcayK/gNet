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
        - TimeDistributed
        - RepeatVector

    Layer should be used with Model Class's add method. Rest of the calculation should be done 
    by NN structure.

    Author : @MGokcayK 
    Create : 04 / 04 / 2020
    Update : 15 / 03 / 2021
                Adding LSTMCell and GRUCell to turn Cell base implementation.
"""

# import required modules
import os
import numpy as np
from gNet import tensor as T
from gNet.activation_functions import __activationFunctionsDecleration as AD
from gNet.initializer import __initializeDeclaretion as ID
import gNet.conv_utils as conv_util
from texttable import Texttable





class Layer:
    """
        Base class of layer implementation.

        Layer shoudl have two methods which are `_init_trainable`, `__call__` and `compute`.
        These methods can be adapted for proper implementation. 
        `_init_trainable` method can be pass  because of layer's need of initialization.
        `__call__` method should be implemented for each layer to connect layers each other.
        `compute` method should be implemented for each layer. Calculation of layer
        done by `compute` method. 

        Base class has also different methods to create easy child class. `_set_initializer`,
        `_get_inits`, and `zero_grad` are helper methods and not called separately by child
        classed. 
    """
    def __init__(self, **kwargs) -> None:
        self._actFuncCaller = AD
        self._trainable = []
        self._layer_name = "Base Layer"
        self._act_name = "Base Activation"
        self._layer_output_shape = 0
        self._numOfParams = 0
        self._layerNo = 0
        self._preLayer = None
        self._nextLayers = []

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
            This method make zero of all layers' trainable parameters' grad values. 
            This is required for each batch. 
        """
        layers = self.get_layers()
        for layer in layers:
            for trainable in layer._trainable:
                trainable.zero_grad()

    def __call__(self, Layer = None):
        """
            `__call__` method is one of the important methods of layer class.
            It connects current layer to `Layer` which should be previous layer.
        """
        raise NotImplementedError     

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        """
            Compute method is base of computation of layer. This is the core of 
            computation. Implementation should be carefully done. 
            Without compute method, layer cannot be called by NN. 
        """
        raise NotImplementedError

    def _connect_layer(self, Layer) -> None:
        """
            Connecting layers. If Layer arguments in `__call__` method is None, it means it 
            it input layer of model. Others are previous layer of current layer. Thus, it 
            should be connected
        """
        if (Layer != None):
            self._preLayer = Layer
            Layer._nextLayers.append(self)
            self._layerNo = Layer._layerNo + 1

    def regularize(self) -> T.Tensor:
        """
            Regularize method is base of computation of regularization of layer. 
            If regularization is not need in that layer like dropout or flatten, 
            return zero. Implementation should be carefully done. 
            Without regularize method, layer cannot be called by NN. 
        """
        raise NotImplementedError

    def get_layers(self) -> list:
        """
            It return all layers which connected from input layer in list.
        """
        layers  = []
        preLayer = True
        rLayer = self # root(input) layer 
        while (preLayer):
            if (rLayer._preLayer != None):
                rLayer = rLayer._preLayer
            else:
                preLayer = False
        
        layers.append(rLayer)
        for item in rLayer._nextLayers:
            self._nL(item, layers)
                
        return layers

    def _nL(self, node, layers) -> None:
        """
            Helper for searching layer in node perspective.
        """
        layers.append(node)
        for nl in node._nextLayers:
            self._nL(nl, layers)

    def save_model(self, file_name='gNet_weights'):
        '''
            Save model parameters of Neural Network w.r.t file name.
            File extension will be `.npy`.

            Argument:
            ---------
                
                file_name           : name of file which store the parameters of NN.
                    >>> type        : string
                    >>> Default     : gNet_weights 
        '''
        _layer = self.get_layers()
        sm = []
        # added each layer's trainable parameters to list 
        for layer in _layer:
            app_item = layer.trainable
            # if layer is Batch Norm. save also running mean and running variance
            if layer._layer_name == 'Batch Normalization':
                app_item = [layer.trainable[0], layer.trainable[1], layer._r_mean, layer._r_var]
            sm.append(app_item)
        # set file name
        fName = file_name + '.npy'
        # save parameters 
        np.save(fName, sm)
        # if everythings passed, print the success.
        if os.path.isfile(fName):
            print('Model weights of `' + fName + '` saved successfully..')

    def load_model(self, file_name='gNet_weights'):
        '''
            Load model parameters of Neural Network from file.
            File extension will be `.npy`.

            Argument:
            ---------
                
                file_name           : name of file which store the parameters of NN.
                    >>> type        : string
                    >>> Default     : gNet_weights 
        '''
        # get model parameters
        _layer = self.get_layers()
        fName = file_name + '.npy'
        w = np.load(fName, allow_pickle=True)
        # check layer properties is same as saved ones.
        for ind, layer in enumerate(_layer):
            for ind_tra, trainable in enumerate(layer.trainable):
                # if layer is Batch Norm. load also running mean and running variance
                if layer._layer_name == 'Batch Normalization':
                    tmp = w[ind][ind_tra]
                    t_shape = tmp.shape
                    layer._r_mean = w[ind][2]
                    layer._r_var = w[ind][3]
                else:
                    t_shape = w[ind][ind_tra].shape

                assert trainable.shape == t_shape, \
                    str('Check ' + layer._layer_name + ' or Layer No:'+str(ind) \
                        +' parameters of model. \n'\
                        'Loaded model are not proper.\n' + \
                        'Model :' + str(trainable.shape) + \
                        '\tLoaded :' +str(t_shape))
            layer.trainable = w[ind]
        # if everythings passed, print the success.
        if os.path.isfile(fName):
            print('Model weights of `' + fName + '` loaded successfully..')

    def get_model_summary(self, show=True, save=False, summary_name='gNet_model_summary.txt', show_pre_layers = False):
        '''
            Get model summary of Neural Network. Summary can be showed, saved or both of them.

            Arguments:
            ---------
                
                show                : show the figure of loss during training.
                    >>> type        : bool
                    >>> Default     : True

                save                : save the figure of loss during training.
                    >>> type        : bool
                    >>> Default     : False
                     
                figure_name         : name of file which store the summary of model.
                    >>> type        : string
                    >>> Default     : gNet_model_summary.txt

                show_pre_layers     : show previous layer information in summary.
                    >>> type        : bool
                    >>> Default     : False
        '''
        # create texttable
        params_no = 0
        t = Texttable()

        t.add_rows([['Layer No (Previous Layer) | Layer', 'Output Shape', '# of Parameters']])
        for layer in self.get_layers():
            if layer._layerNo >0:
                fCol = str(layer._layerNo)+'('+str(layer._preLayer._layerNo)
                if show_pre_layers:
                    fCol += ' ' + layer._preLayer._layer_name 
                fCol += ')' + ' | '+ layer._layer_name
            else:
                fCol = str(layer._layerNo)+ ': '+ layer._layer_name
            tmp = [fCol, layer._layer_output_shape, layer._numOfParams]
            params_no += layer._numOfParams
            t.add_row(tmp)
        t.add_row(['Total', ' ', '{:,}'.format(params_no)])
        if show:
            print(t.draw())
        if save:
            f = open(summary_name, 'w')
            f.write(t.draw())

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
        self._layer_name = "Dense : " + str(self._activation)
        self._act_name = self._activation
        self._layer_output_shape = self._neuronNumber

    def __call__(self, Layer: Layer = None) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        # connect layer to this layer
        self._connect_layer(Layer)

        # if activation function is `str` create caller.
        if isinstance(self._activation, str):
            self._actCaller = self._actFuncCaller[self._activation]()
        else:
            self._actCaller = self._activation

        self._init_trainable()

    def _init_trainable(self):
        '''
            Initialization of dense layer's trainable variables.
        '''
        if self._layerNo == 0:
            row = 1
            col = 1
        else:
            #row = params['layer_output_shape'][self._thisLayer-1]
            row = self._preLayer._layer_output_shape
            if (type(row)==tuple):
                #row = params['layer_output_shape'][self._thisLayer-1][0]
                row = self._preLayer._layer_output_shape[0]
            #col = params['layer_output_shape'][self._thisLayer]
            col = self._layer_output_shape
            if (type(col)==tuple):
                #col = params['layer_output_shape'][self._thisLayer][0]
                col = self._layer_output_shape[0]

        _w_shape = (row, col)
        _b_shape = [col]

        #params['#parameters'].append(row*col+col)
        self._numOfParams = row * col + col

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
        self._layer_name = "flatten"
        self._act_name = "none"
        self._numOfParams = 0

    def __call__(self, Layer: Layer = None) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        # connect layer to this layer
        self._connect_layer(Layer)

        self._neuronNumber = 1
        # If flatten layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            # calculate flattened layer neuron number
            for item in self._input_shape:
                self._neuronNumber *= item
            self._layer_output_shape = self._neuronNumber

        else:
            # we should calculate shape of input. we should know previous layer output shape
            self._neuronNumber = 1
            assert self._layerNo != 0, 'Please make sure that flatten layer is not first layer. \n\
            If it is first layer, give it input_shape.'
            #for item in params['layer_output_shape'][self._thisLayer - 1]:
            for item in self._preLayer._layer_output_shape:
                self._neuronNumber *= item
            self._layer_output_shape = self._neuronNumber

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
        self._act_name = activation_function
        self._layer_name = 'Activation Layer :' + str(self._act_name)

    def __call__(self, Layer: Layer = None) -> None:
        """
            Update some model and class parameters.
        """
        # connect layer to this layer
        self._connect_layer(Layer)

        # assign output shape of layer
        self._layer_output_shape = self._preLayer._layer_output_shape

        # if activation function is `str` create caller.
        if isinstance(self._act_name, str):
            self._actCaller = self._actFuncCaller[self._act_name]()
        else:
            self._actCaller = self._act_name

    def _init_trainable(self):
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
        self._layer_name = "Conv1D"
        self._act_name = "none"

    def __call__(self, Layer: Layer = None) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        # connect layer to this layer
        self._connect_layer(Layer)

        # assign number of parameters of layer
        self._numOfParams = self._filter * self._K + self._filter

        # If Conv1D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            assert len(self._input_shape) == 2, 'Make sure that input of Conv1D has 2 dimension without batch such as (1,28).'
            # get channel, width and height of data
            self._C, self._W = self._input_shape
        else:
            # Conv1D layer is not first layer. So get channel and width 
            # of data from previous layer output shape
            assert self._layerNo != 0, 'First layer of Conv1D should have input_shape!'
            self._C, self._W = self._preLayer._layer_output_shape

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

        self._layer_output_shape = (self._filter, self._O)

        self._init_trainable()

    def _init_trainable(self):
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
        self._layer_name = "Conv2D"
        self._act_name = "none"

    def __call__(self, Layer: Layer = None) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        # connect layer to this layer
        self._connect_layer(Layer)
        
        self._numOfParams = self._filter * self._HH * self._WW + self._filter

        # If Conv2D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            assert len(self._input_shape) == 3, 'Make sure that input of Conv2D has 3 dimension without batch such as (1,28,28).'
            # get channel, width and height of data
            self._C, self._H, self._W = self._input_shape
        else:
            # Conv2D layer is not first layer. So get channel, width and height
            # of data from previous layer output shape
            assert self._layerNo != 0, 'First layer of Conv2D should have input_shape!'
            self._C, self._H, self._W = self._preLayer._layer_output_shape

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

        self._layer_output_shape = (self._filter, self.H_out, self.W_out)

        self._init_trainable()

    def _init_trainable(self):
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
        k, i, j = conv_util.get_im2col_indices(inputs.shape, (self._HH, self._WW), (self._stride_row, self._stride_col), self._layer_output_shape )
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
        self._layer_name = "Conv3D"
        self._act_name = "none"

    def __call__(self, Layer: Layer = None) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        # connect layer to this layer
        self._connect_layer(Layer)

        # If Conv3D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            assert len(self._input_shape) == 4, 'Make sure that input of Conv3D has 4 dimension without batch such as (3,10,28,28).'
            # get channel, width and height of data
            self._C, self._D, self._H, self._W = self._input_shape
        else:
            # Conv3D layer is not first layer. So get channel, depth, width and height
            # of data from previous layer output shape
            assert self._layerNo != 0, 'First layer of Conv3D should have input_shape!'
            self._C, self._D, self._H, self._W = self._preLayer._layer_output_shape

        self._numOfParams = self._filter * self._K_shape[0] * self._K_shape[1] * self._K_shape[2] * self._C + self._filter

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
        
        self._layer_output_shape = (self._filter, self.D_out, self.H_out, self.W_out)

        self._init_trainable()

    def _init_trainable(self):
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
        l, k, i, j = conv_util.get_conv3D_indices(base_shape, self._K_shape, self._stride, self._layer_output_shape)
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
        self._layer_name = "MaxPool1D"
        self._act_name = "none"

    def __call__(self, Layer: Layer = None) -> None:
        # connect layer to this layer        
        self._connect_layer(Layer)

        # If MaxPool1D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            # get channel and width of data
            self._C, self._W = self._input_shape
        else:
            # MaxPool1D layer is not first layer. So get channel and width 
            # of data from previous layer output shape
            self._C, self._W = self._preLayer._layer_output_shape

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

        self._layer_output_shape = (self._C, self._O)

    def _init_trainable(self):
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
        self._layer_name = "MaxPool2D"
        self._act_name = "none"

    def __call__(self, Layer: Layer = None) -> None:
        # connect layer to this layer
        self._connect_layer(Layer)

        # If MaxPool2D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            # get channel, width and height of data
            self._C, self._H, self._W = self._input_shape
        else:
            # MaxPool2D layer is not first layer. So get channel, width and height
            # of data from previous layer output shape
            self._C, self._H, self._W = self._preLayer._layer_output_shape

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

        self._layer_output_shape = (self._C, self.H_out, self.W_out)

    def _init_trainable(self):
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
        k, i, j = conv_util.get_im2col_indices(inputs.shape, (self._HH, self._WW), (self._stride_row, self._stride_col), self._layer_output_shape)
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
        self._layer_name = "MaxPool3D"
        self._act_name = 'none'

    def __call__(self, Layer: Layer = None) -> None:
        # connect layer to this layer
        self._connect_layer(Layer)

        # If MaxPool3D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            # get channel, depth, width and height of data
            self._C, self._D, self._H, self._W = self._input_shape
        else:
            # MaxPool3D layer is not first layer. So get channel, width and height
            # of data from previous layer output shape
            self._C, self._D, self._H, self._W = self._preLayer._layer_output_shape

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

        self._layer_output_shape = (self._C, self.D_out, self.H_out, self.W_out)

    def _init_trainable(self):
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
        l, k, i, j = conv_util.get_conv3D_indices(inputs.shape, self._K_shape, self._stride, self._layer_output_shape)
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
        self._layer_name = "AveragePool1D"
        self._act_name = "none"
        
    def __call__(self, Layer: Layer = None) -> None:
        # connect layer to this layer
        self._connect_layer(Layer)

        # If AveragePool1D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            # get channel and width of data
            self._C, self._W = self._input_shape
        else:
            # AveragePool1D layer is not first layer. So get channel and width 
            # of data from previous layer output shape
            self._C, self._W = self._preLayer._layer_output_shape

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
        
        self._layer_output_shape = (self._C, self._O)

    def _init_trainable(self):
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
        self._layer_name = "AveragePool2D"
        self._act_name = "none"

    def __call__(self, Layer: Layer = None) -> None:
        # connect layer to this layer
        self._connect_layer(Layer)

        # If AveragePool2D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            # get channel, width and height of data
            self._C, self._H, self._W = self._input_shape
        else:
            # AveragePool2D layer is not first layer. So get channel, width and height
            # of data from previous layer output shape
            self._C, self._H, self._W = self._preLayer._layer_output_shape

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

        self._layer_output_shape = (self._C, self.H_out, self.W_out)

    def _init_trainable(self):
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
        k, i, j = conv_util.get_im2col_indices(inputs.shape, (self._HH, self._WW), (self._stride_row, self._stride_col), self._layer_output_shape)
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
        self._layer_name = "AveragePool3D"
        self._act_name = "none"
        
    def __call__(self, Layer: Layer = None) -> None:
        # connect layer to this layer
        self._connect_layer(Layer)

        # If AveragePool3D layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            # get channel, depth, width and height of data
            self._C, self._D, self._H, self._W = self._input_shape
        else:
            # AveragePool3D layer is not first layer. So get channel, width and height
            # of data from previous layer output shape
            self._C, self._D, self._H, self._W = self._preLayer._layer_output_shape

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

        self._layer_output_shape = (self._C, self.D_out, self.H_out, self.W_out)

    def _init_trainable(self):
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
        l, k, i, j = conv_util.get_conv3D_indices(inputs.shape, self._K_shape, self._stride, self._layer_output_shape)
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
        self._layer_name = "Dropout (" + str(self._drop_prob) + ")"
        self._act_name = "none"


    def __call__(self, Layer: Layer = None) -> None:
        # connect layer to this layer
        self._connect_layer(Layer)

        # Dropout layer is not first layer. So output shape from previous layer
        self._layer_output_shape = self._preLayer._layer_output_shape

        # add output shape to model params without batch_size
        #params['layer_output_shape'].append(self._output_shape)
        # add previous layer neuron number to model params
        #params['model_neuron'].append(params['model_neuron'][self._thisLayer - 1])


    def _init_trainable(self):
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
        self._layer_name = "Batch Normalization"
        self._act_name = "none"

    def __call__(self, Layer: Layer = None) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        # connect layer to this layer
        self._connect_layer(Layer)
        self._layer_output_shape = self._preLayer._layer_output_shape
        self._init_trainable()

    def _init_trainable(self):
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
        
        pre_layer_output = np.array([self._preLayer._layer_output_shape])
        
        # if BN after dense layer, parameter shape is different than after conv layers.
        if len(pre_layer_output.shape) == 1:
            self._gamma = self._gamma_init.get_init(shape=pre_layer_output)
            self._beta = self._beta_init.get_init(shape=pre_layer_output)
            self._r_mean = self._r_mean_init.get_init(shape=pre_layer_output)
            self._r_var = self._r_var_init.get_init(shape=pre_layer_output)
            #params['#parameters'].append(pre_layer_output[0] * 4)
            self._numOfParams = pre_layer_output[0] * 4
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
            #params['#parameters'].append(pre_layer_output[0][0] * 4)
            self._numOfParams = pre_layer_output[0][0] * 4

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
        self._layer_name = 'SimpleRNN : ' + str(self._activation)
        self._act_name = str(self._activation)

    def __call__(self, Layer: Layer = None) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        # connect layer to this layer
        self._connect_layer(Layer)

        # If SimpleRNN layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            assert len(self._input_shape) == 2, 'Make sure that input of SimpleRNN has 2 dimension without batch such as (10,5).'
            # get sequencial lenght and input size
            self._seq_len, self._inp_size = self._input_shape
        else:
            # SimpleRNN layer is not first layer. So get sequential lenght and input size 
            # ofrom previous layer output shape
            assert self._layerNo != 0, 'First layer of SimpleRNN should have input_shape!'
            if isinstance(self._preLayer._layer_output_shape, int):
                self._preLayer._layer_output_shape = [self._preLayer._layer_output_shape]
            assert len(self._preLayer._layer_output_shape) == 2, 'Previous RNN layer`s `return_sequences` should be True'
            self._seq_len, self._inp_size = self._preLayer._layer_output_shape

        if self._ret_seq:
            self._layer_output_shape = (self._seq_len, self._cell)
        else:
            self._layer_output_shape = (self._cell)

        self._numOfParams = self._cell * ( self._cell + self._inp_size + 1)

        # if activation function is `str` create caller.
        if isinstance(self._activation, str):
            self._actCaller = self._actFuncCaller[self._activation]()
        else:
            self._actCaller = self._activation

        self._init_trainable()

    def _init_trainable(self):
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



class LSTMCell(Layer):
    """
        LSTMCell layer implementation based on Keras' implementation done by @Author.

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
        super(LSTMCell, self).__init__(**kwargs)
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
        self._act_name = str('(Kernel : ' + str(self._activation)+ ' & Hidden : ' + str(self._hidden_activation) + ')')
        self._layer_name = 'LSTM : ' + self._act_name

    def __call__(self, Layer: Layer = None) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        # connect layer to this layer
        self._connect_layer(Layer)

        # If LSTM layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            assert len(self._input_shape) == 2, 'Make sure that input of LSTM has 2 dimension without batch such as (10,5).'
            # get sequencial lenght and input size
            self._seq_len, self._inp_size = self._input_shape
        else:
            # SimpleRNN layer is not first layer. So get sequential lenght and input size 
            # ofrom previous layer output shape
            assert self._layerNo != 0, 'First layer of LSTM should have input_shape!'
            if isinstance(self._preLayer._layer_output_shape, int):
                self._preLayer._layer_output_shape = [self._preLayer._layer_output_shape]
            assert len(self._preLayer._layer_output_shape) == 2, 'Previous RNN layer`s `return_sequences` should be True'
            self._seq_len, self._inp_size = self._preLayer._layer_output_shape

        if self._ret_seq:
            self._layer_output_shape = (self._seq_len, self._cell)
        else:
            self._layer_output_shape = (self._cell)

        self._numOfParams = 4 * self._cell * ( self._cell + self._inp_size + 1)

        # if activation function is `str` create caller.
        if isinstance(self._activation, str):
            self._actCaller = self._actFuncCaller[self._activation]()
        else:
            self._actCaller = self._activation

        # if hidden activation function is `str` create caller.
        if isinstance(self._hidden_activation, str):
            self._hidden_actCaller = self._actFuncCaller[self._hidden_activation]()
        else:
            self._hidden_actCaller = self._hidden_activation

        self._init_trainable()

    def _init_trainable(self):
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

        
    def computeCell(self, inputs: T.Tensor, h: T.Tensor, cell_state: T.Tensor, **kwargs) -> T.Tensor:
        '''
            Computation of LSTM layer.
        '''
        # finding value to activate
        t_h = h.detach()
        inp = inputs
        f = T.matmul(inp, self._trainable[4]) + T.matmul(t_h , self._trainable[0])
        i = T.matmul(inp, self._trainable[5]) + T.matmul(t_h , self._trainable[1])
        c = T.matmul(inp, self._trainable[6]) + T.matmul(t_h , self._trainable[2])
        o = T.matmul(inp, self._trainable[7]) + T.matmul(t_h , self._trainable[3])
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

        return h, cell_state

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

class LSTM(LSTMCell):
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
        super(LSTM, self).__init__(
            cell,
            activation_function,
            hidden_activation_function,
            initializer,
            hidden_initializer,
            bias_initializer,
            return_sequences,
            return_state,
            kernel_regularizer,
            hidden_regularizer,
            bias_regularizer,
            use_bias,
            use_forget_bias,
            input_shape,
            **kwargs)

    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        # initializer hidden matrix and cell state as zeros
        h_init = ID['zeros_init']()
        c_init = ID['zeros_init']()
        h = T.Tensor(h_init.get_init((inputs.shape[0],self._cell)), True)
        cell_state = T.Tensor(c_init.get_init((inputs.shape[0],self._cell)))

        # sequential output holder        
        return_seq = T.Tensor(np.empty((inputs.shape[0],self._cell)))

        # for each sequencial data 
        for s in range(self._seq_len):
            h, cell_state = self.computeCell(inputs[:,s,:], h, cell_state)

            #add sequential output
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



class GRUCell(Layer):
    """
        GRUCell layer implementation based on Keras' implementation done by @Author.

        
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
        super(GRUCell, self).__init__(**kwargs)
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
        self._act_name = str('(Kernel : ' + str(self._activation)+ ' & Hidden : ' + str(self._hidden_activation) + ')')
        self._layer_name = 'GRU : ' + self._act_name

    def __call__(self, Layer: Layer = None) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        # connect layer to this layer
        self._connect_layer(Layer)

        # If GRU layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            assert len(self._input_shape) == 2, 'Make sure that input of GRU has 2 dimension without batch such as (10,5).'
            # get sequencial lenght and input size
            self._seq_len, self._inp_size = self._input_shape
        else:
            # SimpleRNN layer is not first layer. So get sequential lenght and input size 
            # ofrom previous layer output shape
            assert self._layerNo != 0, 'First layer of GRU should have input_shape!'
            if isinstance(self._preLayer._layer_output_shape, int):
                self._preLayer._layer_output_shape = [self._preLayer._layer_output_shape]
            assert len(self._preLayer._layer_output_shape) == 2, 'Previous RNN layer`s `return_sequences` should be True'
            self._seq_len, self._inp_size = self._preLayer._layer_output_shape

        if self._ret_seq:
            self._layer_output_shape = (self._seq_len, self._cell)
        else:
            self._layer_output_shape = (self._cell)

        self._numOfParams = 3 * self._cell * ( self._cell + self._inp_size + 1)

        # if activation function is `str` create caller.
        if isinstance(self._activation, str):
            self._actCaller = self._actFuncCaller[self._activation]()
        else:
            self._actCaller = self._activation

        # if hidden activation function is `str` create caller.
        if isinstance(self._hidden_activation, str):
            self._hidden_actCaller = self._actFuncCaller[self._hidden_activation]()
        else:
            self._hidden_actCaller = self._hidden_activation

        self._init_trainable()

    def _init_trainable(self):
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

        
    def computeCell(self, inputs: T.Tensor, h: T.Tensor, **kwargs) -> T.Tensor:
        '''
            Computation of GRU layer.
        '''
        # finding value to activate
        t_h = h.detach()
        z = T.matmul(inputs, self._trainable[3])  + T.matmul(t_h, self._trainable[0])
        r = T.matmul(inputs, self._trainable[4])  + T.matmul(t_h, self._trainable[1])
        # adding if use_bias is true 
        if self._bias:
            z += self._trainable[6]
            r += self._trainable[7]
        
        # activate value
        z = self._hidden_actCaller.activate(z)
        r = self._hidden_actCaller.activate(r)
        ht = T.matmul(inputs, self._trainable[5]) + T.matmul(r * t_h, self._trainable[2])
        if self._bias:
            ht += self._trainable[8]
        ht = self._actCaller.activate(ht)
        
        # calculate hidden output
        h = z * h + (1-z) * ht

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


class GRU(GRUCell):
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
            >>> Default             : 'sigmoid'

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
        super(GRU, self).__init__(
            cell,
            activation_function,
            hidden_activation_function,
            initializer,
            hidden_initializer,
            bias_initializer,
            return_sequences,
            return_state,
            kernel_regularizer,
            hidden_regularizer,
            bias_regularizer,
            use_bias,
            input_shape,
            **kwargs)
        
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
            h = self.computeCell(inputs[:,s,:], h)

        if self._ret_seq:
            return_seq = T.reshape(return_seq, (self._seq_len,-1, self._cell))
            if self._ret_sta:
                return T.transpose(return_seq, (1,0,2)), h 
            return T.transpose(return_seq, (1,0,2))
        else:
            return h



class TimeDistributed(Layer):
    """
        TimeDistributed layer implementation based on Keras' implementation done by @Author.

        Second dimension is time dimension. If TD layer is used as first layer, input_shape 
        should include time dimension. Else, input_shape is taken as second dimension of previous
        layer of model automatically.

        For example : \n
        input = np.random.randn(32,10,20) # input has 32 batch, 10 time, 20 feature.
        ...
        model.add(gNet.layer.TimeDistributed(gNet.layer.LSTM(5,input_shape=(10,20))) # layer output will be (32,10,5).
        ...

        \n

        Example 2 : \n
        input = np.random.randn(32,10,1,28,28) # input has 32 batch, 10 time, 1 channel ,28 height, 28 width.
        ...
        model.add(gNet.layer.TimeDistributed(gNet.layer.Conv2D(16,input_shape=(10,1,28,28))) # layer output will be (32,10,16,28,28).
        ...

        Arguments for initialization :
        ------------------------------

        layer                       : Time Distributed layer.
            >>> type                : gNet.layer
            >>> Default             : None

        input_shape                 : If TimeDistributed layer is first layer of model, input_shape should be declared.
                                    Shape will be in form of (time, input_shape of layer argument) which has no batch
                                    dimenstion.
            >>> type                : tuple
            >>> Default             : None

        Arguments for compute method is tensor of previous method in proper size.
    """
    def __init__(self,
                layer,
                input_shape = None,
                **kwargs):
        super(TimeDistributed, self).__init__(**kwargs)
        self._time_layer = layer
        assert isinstance(self._time_layer, Layer), "Input layer of TimeDistributed is not the layer of gNet!"
        self._input_shape = input_shape
        
    def __call__(self, Layer: Layer = None) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        # connect layer to this layer
        self._connect_layer(Layer)
        
        # If TD layer is the first layer of model, input_shape should be declared.
        if self._input_shape != None:
            assert len(self._input_shape) >= 2, 'Make sure that input of TimeDistributed has minimum 2 dimension without batch such as (10,5).'
        else:
            # If TD layer is not first layer, get input shape from previous layer.
            self._input_shape = self._preLayer._layer_output_shape

        # calculate basic properties of self._time_layer
        self._time_layer(Layer)
        
        self._act_name = str('TimeDistributed : ' + str(self._time_layer._layer_name))
        
        self._layer_name = self._act_name
                
        # output shape
        self._layer_output_shape = []
        self._layer_output_shape.append(self._input_shape[0])
        if (type(self._time_layer._layer_output_shape) == int):
            self._layer_output_shape.append(self._time_layer._layer_output_shape)
        else:
            [self._layer_output_shape.append(item) for item in self._time_layer._layer_output_shape]
        
        self._layer_output_shape = tuple(self._layer_output_shape)

        # add number of parameters 
        self._numOfParams =  self._time_layer._numOfParams

        # passing trainables of self._time_layer to TD layer to update and optimize.
        [self._trainable.append(trainable) for trainable in self._time_layer._trainable]

    def _init_trainable(self):
        '''
            Initialization of TD layer's trainable variables.
        '''
        pass

        
    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of TD layer.
        '''
        # base shape of input
        base_shape = inputs.shape

        # index of inputs 
        ind = [] 
        [ind.append(slice(0,d)) for d in base_shape]
        ind[1] = 0

        # time dimension computation of self._time_layer
        temp = self._time_layer.compute(inputs[tuple(ind)], train)
        for i in range(base_shape[1]-1):
            ind[1] = i+1 # dynamic index of time dimension
            cout = self._time_layer.compute(inputs[tuple(ind)], train)
            temp = T.append(temp, cout, 1)            

        # final shape of output
        final_shape = []
        final_shape.append(base_shape[0])
        [final_shape.append(item) for item in self._layer_output_shape]

        # reshape output
        inputs = T.reshape(temp, final_shape)
        return inputs


    def regularize(self) -> T.Tensor:
        """
            Regularization of layer. This layer do not have regularizable parameter.
        """
        return T.Tensor(0.)



class RepeatVector(Layer):
    """
        RepeatVector layer implementation done by @Author.

        Repeat output of previous layer in time dimension which is second dimension. 

        For example : \n
        input = np.random.randn(32,1,20) # input has 32 batch, 1 time, 20 feature.
        ...
        model.add(gNet.layer.RepeatVector(5)) # layer output will be (32,5,20).
        ...

        Arguments for initialization :
        ------------------------------

        repeat_number               : Number of repeat.
            >>> type                : int
            >>> Default             : 1

        Arguments for compute method is tensor of previous method in proper size.
    """
    def __init__(self,
                repeat_number = 1,
                **kwargs):
        super(RepeatVector, self).__init__(**kwargs)
        assert type(repeat_number) == int, "repeat_number should be integer!"
        self._repeat = repeat_number
        self._layer_name = str('RepeatVector : ' + str(self._repeat))
        self._act_name = "none"
        
    def __call__(self, Layer: Layer = None) -> None:
        '''
            Update of some of model parameters and class parameters.
        '''
        # connect layer to this layer
        self._connect_layer(Layer)
                
        # output shape
        self._layer_output_shape = []
        self._layer_output_shape.append(self._repeat)
        if (type(self._preLayer._layer_output_shape) == int):
            self._layer_output_shape.append(self._preLayer._layer_output_shape)
        else:
            [self._layer_output_shape.append(item) for item in self._preLayer._layer_output_shape]
        
        self._layer_output_shape = tuple(self._layer_output_shape)

    def _init_trainable(self):
        '''
            Initialization of RepeatVector layer's trainable variables.
        '''
        pass

        
    def compute(self, inputs: T.Tensor, train: bool, **kwargs) -> T.Tensor:
        '''
            Computation of RepeatVector layer.
        '''
        # index of inputs 
        ind = [] 
        [ind.append(d) for d in inputs.shape]
        ind.insert(1, 1)

        # reshape input 
        inputs = T.reshape(inputs, ind)
        
        # repeat inputs 
        temp = inputs
        for i in range(self._repeat-1):
            temp = T.append(temp, inputs, 1)

        return temp


    def regularize(self) -> T.Tensor:
        """
            Regularization of layer. This layer do not have regularizable parameter.
        """
        return T.Tensor(0.)




