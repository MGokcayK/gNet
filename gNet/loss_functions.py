    """
    Loss functions module of gNet.

    Containing Loss functions (with calling string): \n
        - Categorical Cross Entropy ('categoricalcrossentropy', 'cce')
        - Binary Cross Entropy ('binarycrossentropy','bce')
        - Mean Square Error ('meansquareerror', 'mse')

    To call loss function for training, user have two way. \n
        - User can define loss function as string in calling NeuralNetwork.setup() function.
            
            Ex: 
            >>> NeuralNetwork.setup(loss_function='categoricalcrossentropy', optimizer='SGD')

        - User can define loss function explicitly.
            
            Ex:
            >>> loss = gNet.loss_functions.CategoricalCrossEntropy(from_logits=True)
                NeuralNetwork.setup(loss_function=loss, optimizer='SGD')  

    Author : @MGokcayK 
    Create : 25 / 03 / 2020
    Update : 26 / 12 / 2020
                Adding REGISTER_LOSS_FUNCTION function.
"""

import numpy as np
import gNet.tensor as T
import gNet.activation_functions as actFunc
import gNet.metric as mt

__lossFunctionsDecleration = {}

class Loss:
    '''
        Base class of Loss functions implementation.

        Containing Loss functions (with calling string): \n
            - Categorical Cross Entropy ('categoricalcrossentropy', 'cce')
            - Binary Cross Entropy ('binarycrossentropy','bce')
            - Mean Square Error ('meansquareerror', 'mse')

        To call loss function for training, user have two way. \n
            - User can define loss function as string in calling NeuralNetwork.setup() function.
                
                Ex: 
                >>> NeuralNetwork.setup(loss_function='categoricalcrossentropy', optimizer='SGD')

            - User can define loss function explicitly.
                
                Ex:
                >>> loss = gNet.loss_functions.CategoricalCrossEntropy(from_logits=True)
                    NeuralNetwork.setup(loss_function=loss, optimizer='SGD')  

        To implement new loss function, developer need to add loss funtion which
        calculates the loss.

        Also, developer need to add get metric function to class. It return proper
        metric methods.

        Also, developer need to add class to __lossFunctionsDecleration for 
        getting proper loss by their calling string like Categorical Cross 
        Entropy as 'cce' or 'categoricalcrossentropy'.
    '''
    def __init__(self, from_logits=False, epsilon=1e-7, axis=-1, **kwargs) -> None:
        self._eps = epsilon
        self._from_logits = from_logits
        self._axis = axis

    def loss(self, y_true, y_pred, output_layer) -> None:
        raise NotImplementedError

    def get_metric(self) -> mt.Metric:
        raise NotImplementedError

class CategoricalCrossEntropy(Loss):
    '''
        Categorical Cross Entropy (CCE) class.

        Loss = -SUM( yi * log (pi))
                where yi is target and pi is prediction values from neural network.

        For CCE, when argument from_logits=True it means that if last layer
        activation function is not `Softmax`, it will pass the pi to `Softmax` then
        calculate Loss w.r.t.
        Also, if last layer activation function is not `Softmax`, pi will be normalize to
        calculate loss.

        Generally CCE used for more than 2 output neuron.
    '''
    def loss(self, y_true, y_pred, output_layer) -> 'Tensor':
        # make sure that true and prediction values are tensor
        y_true = T.make_tensor(y_true)
        y_pred = T.make_tensor(y_pred)

        # make same shape ones `tensor`.
        eps = T.ones(y_pred.shape) * self._eps

        if (output_layer._act_name != 'softmax'):
            # We need to sure that last layer SHOULD NOT be softmax.
            # If it is softmax, we do not need to check from_logits because 
            # if from_logits == True, we have to apply softmax. It is contradiction.
            if self._from_logits:
                # from_logits means that before calculate loss, we need to normalize
                # output w.r.t softmax. If last layer is softmax, it will be second usage
                # of softmax function. Therefore, it will be contradiction.
                y_pred = actFunc.Softmax.activate(y_pred)
            else:
                # We normalize output without softmax.
                y_pred /= T.tensor_sum(y_pred + eps, self._axis, keepdim=True)   

        # clip prediction values to get rid of infinities
        y_pred.value = np.clip(y_pred.value, self._eps, 1. - self._eps)
        # calculate loss for all samples
        value = -T.tensor_sum(y_true * T.log(y_pred), self._axis)
        # calculate avarage loss of batch
        value = T.mean(value)
        return value

    def get_metric(self) -> mt.Metric:
        """
            Categorical Cross Entropy use Categorical Accuracy for calculate it properly.
        """
        return mt.CategoricalAccuracy()

class BinaryCrossEntropy(Loss):
    '''
        Binary Cross Entropy (BCE) class.

        Loss = -SUM( yi * log (pi)) = -yi * log(pi) - (1 - yi) * log (1 - pi)
                where yi is target and pi is prediction values from neural network.

        For BCE, when argument from_logits=True it means that if last layer
        activation function is not `Sigmoid`, it will pass the pi to `Sigmoid` then
        calculate Loss w.r.t.
        
        Generally it is used for binary classification problems.
    '''
    def loss(self, y_true, y_pred, output_layer):         
        # make sure that true and prediction values are tensor
        y_true = T.make_tensor(y_true)
        y_pred = T.make_tensor(y_pred)

        # make same shape ones `tensor`.
        ones = T.ones(y_pred.shape)
        zeros = ones * 0.

        if (output_layer._act_name != 'sigmoid'):
            # We need to sure that last layer SHOULD NOT be sigmoid.
            # If it is sigmoid, we do not need to check from_logits because 
            # if from_logits == True, we have to apply sigmoid. It is contradiction.
            if self._from_logits:
                # from_logits means that before calculate loss, we need to normalize
                # output w.r.t sigmoid. If last layer is sigmoid, it will be second usage
                # of sigmoid function. Therefore, it will be contradiction.
                cond = (y_pred.value >= zeros)
                pos_y_pred = T.where(y_pred, cond, y_pred, zeros)
                neg_y_pred = T.where(y_pred, cond, -y_pred, y_pred)
                return T.mean( pos_y_pred - y_pred * y_true + \
                                T.log(ones + T.exp(neg_y_pred)))
            else:            
                y_pred.value = np.clip(y_pred.value, self._eps , 1. - self._eps)
                eps = ones * self._eps
                bce = y_true * T.log(y_pred + eps)  
                bce += (ones - y_true) * T.log(ones - y_pred + eps)              
                return T.mean(-bce)

    def get_metric(self) -> mt.Metric:
        """
            Binary Cross Entropy use Binary Accuracy for calculate it properly.
        """
        return mt.BinaryAccuracy()


class MeanSquareError(Loss):
    '''
        Mean Square Error.

        Basic error calculation method based on difference of prediction and targets.

        Loss = SUM((pi - yi)**2) / n 
                where yi is target and pi is prediction values from neural network.
                where n is batch size.
    '''
    def loss(self, y_true, y_pred, output_layer):  
        # make sure that true and prediction values are tensor
        y_true = T.make_tensor(y_true)
        y_pred = T.make_tensor(y_pred)
        error = y_pred - y_true
        error = error ** 2
        return T.mean(T.tensor_sum(error, axis=-1), axis=-1)

    def get_metric(self) -> mt.Metric:
        """
            Mean Square Error use Categorical Accuracy for calculate it properly.
        """
        return mt.CategoricalAccuracy()

def REGISTER_LOSS_FUNCTION(loss_function : Loss, call_name : str):
    """
        Register Loss Function w.r.t `call_name`. 

        Arguments :
        -----------

        loss_function   : Loss function class.
        >>>    type     : gNet.loss_function.Loss()

        call_name       : Calling name of loss function. It will be lowercase. It is not sensible.
        >>>    type     : str
    """
    __lossFunctionsDecleration.update({call_name.lower() : loss_function})


REGISTER_LOSS_FUNCTION(CategoricalCrossEntropy, 'categoricalcrossentropy')
REGISTER_LOSS_FUNCTION(CategoricalCrossEntropy, 'cce')
REGISTER_LOSS_FUNCTION(BinaryCrossEntropy, 'binarycrossentropy')
REGISTER_LOSS_FUNCTION(BinaryCrossEntropy, 'bce')
REGISTER_LOSS_FUNCTION(MeanSquareError, 'meansquareerror')
REGISTER_LOSS_FUNCTION(MeanSquareError, 'mse')