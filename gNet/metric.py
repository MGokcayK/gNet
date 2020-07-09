"""
    Metrics module of gNet.

    Containing metric class : \n
        - CategoricalAccuracy
        - BinaryAccuracy

    This classes calculate accuracy of model w.r.t metric type.

    Author : @MGokcayK 
    Create : 03 / 04 / 2020
    Update : 02 / 06 / 2020
                Added descriptions and BinaryAccuracy.
"""

# import related modules 
import numpy as np
from gNet import tensor as T

class Metric:
    '''
        Base class of Metric.
        
        To implement new metric method, developer need to declare accuracy function which
        calculate accuracy.
    '''
    def __init__(self, **kwargs) -> None:
        self._count = 0.
        self._total = 0.

    def accuracy(self, y_true, y_pred) -> None:
        """
            Calculation of accuracy. Depends on metrics, it should be arranged. 
            Without `accuracy` method, metric class cannot calculate accuracy.

            Formula of accuracy : 

                        Accuracy = counted / total. 

            Counted means that sample which is equal to wanted condition. Condition
            depends on accuracy type.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
            Resetting accuracy.
        """
        self._count = 0.
        self._total = 0.


class CategoricalAccuracy(Metric):
    """
        Categorical Accuracy is calculation of accuracy in Categorical conditions. 
        It compares maximum argument of y_true and y_pred. If they are equal each 
        other counter increased, else stay same. Also, total number of sample 
        counted separately. 
    """
    def accuracy(self, y_true, y_pred) -> float:
        # find maximum values indexes
        argmax_true = np.argmax(y_true.value, axis=-1).reshape(-1,1)
        argmax_pred = np.argmax(y_pred.value, axis=-1).reshape(-1,1)
        # check whether max indexes are equal. 
        # if equal add to count
        self._count += np.equal(argmax_true, argmax_pred).sum()
        # add how many item does validate
        self._total += argmax_pred.shape[0]
        return self._count / self._total

class BinaryAccuracy(Metric):
    """
        Binary Accuracy is calculation of accuracy in Binary conditions. 
        It checks whether prediction is higher than threshold. If pred is higher,
        it compares maximum argument of y_true and y_pred. If they are equal each 
        other counter increased, else stay same. Also, total number of sample 
        counted separately. 
    """
    def __init__(self, threshold=0.5) -> None: 
        self._threshold = threshold

    def accuracy(self, y_true, y_pred) -> float:
        # set values which are bigger than threshold is 1.
        argmax_pred = np.where(y_pred.value > self._threshold, 1., 0.)
        # find maximum values indexes
        argmax_true = np.argmax(y_true.value, axis=-1).reshape(-1,1)
        argmax_pred = np.argmax(argmax_pred, axis=-1).reshape(-1,1)
        # check whether max indexes are equal. 
        # if equal add to count
        self._count += np.equal(argmax_true, argmax_pred).sum()
        # add how many item does validate
        self._total += argmax_pred.shape[0]
        return self._count / self._total   

