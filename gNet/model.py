"""
    Model module of gNet.

    Model module has Model class which create models by adding layers and 
    store model parameters.   

    Author : @MGokcayK 
    Create : 04 / 04 / 2020
    Update : 23 / 12 / 2020
                Change `add` method for functional layer connection properties of gNet.
"""
import numpy as np
from typing import List
from gNet import tensor as T
from gNet.layer import Layer
from gNet.initializer import __initializeDeclaretion as ID


class Model:
    """
        Model class implementation.

        This class create a model by adding layers sequentially and 
        store layers' parameters. 
    """
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            'layer_number': 0,
            'layer_name' : [],
            'activation' : [],
            'layers' : [], 
            'layer_output_shape' : [],
            '#parameters' : []
        }
        print('\n\nModel created and initializing parameters..')

    def add(self, Layer : Layer) -> None:
        """
            Layer addition method. 
            
            It adds layer by storing them into params dict 
            with 'layers' tag. Also, it initialize all layer by 
            calling them with __call__ magic methods (which 
            implemented in layer class.).
        """
        if (self._params['layer_number'] != 0):
            Layer(self._params['layers'][self._params['layer_number']-1])
        else:
            Layer()
        
        self._params['layers'].append(Layer)        
        self._params['layer_name'].append(self._params['layers'][self._params['layer_number']]._layer_name)
        self._params['activation'].append(self._params['layers'][self._params['layer_number']]._act_name)
        self._params['#parameters'].append(self._params['layers'][self._params['layer_number']]._numOfParams)
        self._params['layer_output_shape'].append(self._params['layers'][self._params['layer_number']]._layer_output_shape)

        self._params['layer_number'] += 1

    def get_layers(self) -> List:
        """
            Getting layers of model.
        """
        return self._params['layers']

    def zero_grad(self) -> None:
        """
            Zeroing trainable variables of model.
        """
        self._params['layers'][-1].zero_grad()

    def get_params(self) -> None:
        """
            Getting parameters of model.
        """
        return self._params