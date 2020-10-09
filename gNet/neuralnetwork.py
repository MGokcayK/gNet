"""
    Neural Network module of gNet.

    Containing Neural Network structure is Supervised Learning's structure named as NeuralNetwork.

    NeuralNetwork (NN) has only 1 input which is model. Model should created by Model module.
    The model can be only MLP model or CNN (ConvNN) model. Both of model can be accepted.

    NN have all base functions which are `train`, `setup`, `predict`, `evaluate`, `save_model`,
    `load_model`, `get_loss_plot`, `get_accuracy_plot` and `get_model_summary`. 
    All of these functions makes base of gNet.

    To create NN, class should be construct first, then select the options.

    Author : @MGokcayK 
    Create : 25 / 03 / 2020
    Update : 09 / 10 / 2020
                Add new functionalities to get_loss_plot, get_accuracy_plot and evaluate methods.
"""

# import built in modules
import os
import time 
import datetime

# import gNet modules and functions
from gNet import tensor as T
from gNet import metric as mt
from gNet.optimizer import __optimizerDecleration as OD
from gNet.initializer import __initializeDeclaretion as ID
from gNet.loss_functions import __lossFunctionsDecleration as LD

# import 3rd party modules and functions
import numpy as np
from texttable import Texttable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class NeuralNetwork:
    '''
        Neural Network module of gNet.

        Containing Neural Network structure is Supervised Learning's structure named as NeuralNetwork.

        Arguments:
        ----------

            model : model of NN.

        NeuralNetwork (NN) has only 1 input which is model. Model should created by Model module.
        The model can be only MLP model or CNN (ConvNN) model. Both of model can be accepted.

        NN have all base functions which are `train`, `setup`, `predict`, `evaluate`, `save_model`,
        `load_model`, `get_loss_plot`, `get_accuracy_plot` and `get_model_summary`. 
        All of these functions makes base of gNet.

        To train or evalute, model shoudl have setup. Therefore, user should call setup function first.

        To create NN, class should be construct first, then select the options.
        For example : to create MNIST model to train and evaluate;

        ### import modules of gNet
        >>> from gNet import neuralnetwork as NN
        >>> from gNet import model as mdl
        >>> from gNet import layer as lyr
        >>> from gNet import optimizer as optim
        # create model
        >>> model = mld.Model()
        # add layers 
        >>> model.add(lyr.Flatten(input_shape=(1,28,28)))
        >>> model.add(lyr.Dense(128,'relu'))
        >>> model.add(lyr.Dense(10,'softmax'))
        # create NN
        >>> net = NN.NeuralNetwork(model)
        # to get and save model summary 
        >>> net.get_model_summary(save=True)
        # to select optimizer and adjust learning rate 
        >>> opt = optim.Adam(lr=0.0001)
        # to setup NN properties
        >>> net.setup(loss_function='cce', optimizer=opt)
        # to train NN
        >>> net.train(x_train, y_train, batch_size=32, epoch=10)
        # to evaluate NN
        >>> net.evaluate(x_test,y_test)
        # to predict
        >>> net.predict(x_pre)
        # to get and save loss plot
        >>> net.get_loss_plot(show=True, save=True)
        # to get and save accuracy plot
        >>> net.get_accuracy_plot(show=True, save=True)
        # to save model parameters/weights
        >>> net.save_model()
        # to load model parameters/weights
        >>> net.load_model()
    '''

    def __init__(self, model):
        ''' Initialize the Neural Network.'''
        # set loss function caller to select dictionary w.r.t strings
        self._lossFuncCaller = LD
        # set optimizer caller to select dictionary w.r.t strings
        self._optimizerCaller = OD
        # set model
        self._model = model
        # set layers of model
        self._layer = model.get_layers()
        # initialize history of loss and accuracy which used by train and train_one_batch 
        self.his_loss = []
        self.his_acc = []
        # initialize some parameters for one_batch methods
        self._ite= 0
        self._epoch_loss = 0
        self._val_epoch_loss = 0
        self._val_ite = 0
        self._eva_loss = 0
        self._eva_ite = 0
        self._eva_acc_reset = False
        self._printing_one_batch = None
        # create flag for training type
        self.TRAIN_ONE_BATCH_FLAG = False
        self.VALIDATION_FLAG = False


    def _feedForward(self, inp, train=False):
        ''' Feed Forward of NN w.r.t input as inp.'''
        # for each layer, compute the layer and pass the parametes as inp to next layer
        for layer in self._layer:
            #print(inp.shape, layer)
            inp = layer.compute(inp, train)
        return inp


    def _regularizer(self):
        ''' Regularization of NN.'''
        # for each layer, compute the regularization and sum them up.
        _res = 0.
        for layer in self._layer:
            _res += layer.regularize()
        return _res


    def _train_print(self, end_epoch=False, percentage=None, printing_var=None):
        """
            Printing some parameters during training. 

            If printing at the end of epoch, make end_epoch = True to print new line, else it will print only 1 lines during whole training.

            If percentage is given, it will print it. It should be float.

            printing operation depends on which strings in printing_var. For :
                - Loss                  : 'loss',
                - Epoch loss            : 'epoch_loss',
                - Accuracy              : 'accuracy',
                - Validation Loss       : 'val_loss',
                - Validation Accuracy   : 'val_acc',
                - ETA                   : 'ETA' 
                will be in printing_var.

            printing_var is list of strings for printing parameters.
        """
        printing_list = ''
        if not self.TRAIN_ONE_BATCH_FLAG:
            printing_list = 'Epoch : %d / %d ' % (self._e + 1, self._epoch)
        
        if percentage:
            # arrange print or percentage
            if percentage > 100:
                percentage = 100
            printing_list += '  %.2f %% ' % (percentage)

        if 'loss' in printing_var:
            printing_list += ' Loss : %.4f ' % (np.round(self._epoch_loss / self._ite, 4))

        if 'epoch_loss' in printing_var:
            printing_list += ' Epoch loss : %.4f ' % (np.round(self._epoch_loss , 4))

        if 'accuracy' in printing_var:
            printing_list += ' Accuracy : %.4f ' % (np.round(self._accVal, 4))

        # if validation print validation parameters.
        if self.VALIDATION_FLAG:
            if 'val_loss' in printing_var:
                printing_list += ' Val. Loss : %.4f ' % (np.round(self._val_epoch_loss / self._val_ite, 4))
        
            if 'val_acc' in printing_var:
                printing_list += ' Val. Accuracy : %.4f ' % (np.round(self._val_accVal, 4))
            # reset printing for validation.
            self.VALIDATION_FLAG = False

        if 'ETA' in printing_var:
            if self.TRAIN_ONE_BATCH_FLAG:
                raise ValueError('ETA is not calculate for one batch!')
            else:
                printing_list += ' ETA : ' + str(datetime.timedelta(seconds=np.ceil(self._pr_eta)))

        # priniting at the end of epoch is different in epoch. Difference is updating on line or not.
        if end_epoch:
            print(printing_list)
        else:
            print(printing_list, end='\r')


    def train(self, x, y, batch_size=32, epochs=1, val_x=None, val_y=None, val_rate=None, printing=['loss', 'accuracy'], shuffle=True):
        '''
            Train function of Neural Network structure. To train NN, this function should
            be called. 

            Arguments:
            ----------
                
                x               : train data.

                y               : label data of train.

                batch_size      : batch size of NN.
                    >>> type    : int
                    >>> Default : 32

                epoch           : epoch number of train.
                    >>> type    : int
                    >>> Default : 1

                val_x           : validation data.
                    
                val_y           : validation label data.

                val_rate        : validation rate of given x data and value between [0,1)
                    >>> type    : float
                    >>> Default : None (0.0 is same as None)

                printing        : printing parameters during training.
                    >>> type    : list
                    >>> Default : ['loss', 'accuracy'] 

                shuffle         : shuffle data for each epoch.
                    >>> type    : bool
                    >>> Default : True
            
        '''
        print('Train starting..')
        # create start time to calculate total time for training.
        s_time = time.time()
        # set batch size for class
        self._batchSize = batch_size
        # set epoch for class
        self._epoch = epochs

        # if validation rate is not None, split that much data as validation.
        if val_rate:
            assert val_rate != 1.0, 'Validation rate should be less than 1.0 and greater equat than 0.'
            # check whether val_x and val_y is given with val_rate at the same time. If it is given, warn the user.
            assert val_x is None and val_y is None, 'Validation rate and validation inputs are not used at the same time. '
            # find how many data for validation
            val_ind = int(x.shape[0] * val_rate)
            # shuffle indexing of data to prepare validation then shuffle
            shuff_ind = np.arange(0, x.shape[0])
            np.random.shuffle(shuff_ind)
            x = x[shuff_ind]
            y = y[shuff_ind]
            # split data to validation and trainig.
            val_y = y[0:val_ind]
            val_x = x[0:val_ind]
            x = x[val_ind:]
            y = y[val_ind:]     
            
        for e in range(self._epoch):
            self._e = e
            # get start index of data for each epoch.
            self._starts = np.arange(0, x.shape[0], self._batchSize)
            if val_x is not None and val_y is not None:
                self._val_starts = np.arange(0, val_x.shape[0], self._batchSize)
            # if data willing to shuffled, shuffle it by shuffling index for each epoch.
            if shuffle:
                np.random.shuffle(self._starts)
                if val_x is not None and val_y is not None:
                    np.random.shuffle(self._val_starts)

            # initialize percentage and batch counter for each epoch.
            self._per= 0
            self._b_cnt = 0
            # initialize iteration and epoch loss for each epoch
            self._ite= 0
            self._epoch_loss = 0
            self._val_ite = 0
            self._val_epoch_loss = 0
            # reset accuracy for each epoch
            self._acc.reset()          
            self._val_acc.reset()   

            # run batchs 
            for _start in self._starts:
                # find last index of batch and iterate other parameters.
                _end = _start + self._batchSize
                self._per += self._batchSize
                self._ite += 1
                self._b_cnt += 1
                # initialize ETA time to estimae each epoch time.
                self._eta_time = time.time()

                # make zero of all grads of model because of AD structure.
                self._model.zero_grad()

                # get batch of train and label data                          
                _x_batch = T.make_tensor(x[_start:_end])
                _y_batch = T.make_tensor(y[_start:_end])

                # predict the batch
                _pred = self._feedForward(_x_batch, True)
                # calculate loss, grad, epoch loss which is total loss of epoch and accuracy
                _loss = self._loss.loss(_y_batch, _pred, self._model.get_params()) + self._regularizer()
                _loss.backward()
                self._epoch_loss += np.mean(_loss.value)
                self._accVal = self._acc.accuracy(_y_batch,_pred)

                # append history of loss and accuracy
                self.his_loss.append(np.round(self._epoch_loss / self._ite, 4))
                self.his_acc.append(np.round(self._accVal, 4))

                # optimize layer parameters
                self._optimizer.step(self._layer)

                # calculate ETA for epoch
                self._pr_eta = round((time.time()-self._eta_time),2) * (len(self._starts) - self._b_cnt)
                # print properties
                self._train_print(percentage=round((self._per / x.shape[0] * 100), 2), printing_var=printing)

            # validation same as traning, difference is not calling backpropagation and optimizer
            if val_x is not None and val_y is not None:
                self.VALIDATION_FLAG = True
                for _val_start in self._val_starts:
                    _val_end = _val_start + self._batchSize
                    _val_x_batch = T.make_tensor(val_x[_val_start:_val_end])
                    _val_y_batch = T.make_tensor(val_y[_val_start:_val_end])                    
                    _val_pred = self._feedForward(_val_x_batch)
                    _val_loss = self._loss.loss(_val_y_batch, _val_pred, self._model.get_params())
                    self._val_epoch_loss +=  np.mean(_val_loss.value)
                    self._val_accVal = self._val_acc.accuracy(_val_y_batch,_val_pred)
                    self._val_ite += 1

            # print properties
            self._train_print(end_epoch=True, percentage=round((self._per / x.shape[0] * 100), 2), printing_var=printing)

        # print training time
        print("Passed Training Time : ", datetime.timedelta(seconds=(time.time()-s_time)))


    def train_one_batch(self, x_batch, y_batch, val_x=None, val_y=None, val_rate=None, printing=['loss', 'accuracy'], single_batch=True):
        '''
            Train single batch of data. 

            Arguments:
            ----------
                
                x_batch         : train data of batch.

                y_batch         : label data of batch.

                val_x           : validation data.
                    
                val_y           : validation label data.

                val_rate        : validation rate of given x data and value between [0,1)
                    >>> type    : float
                    >>> Default : None (0.0 is same as None)

                printing        : printing parameters during training.
                    >>> type    : list
                    >>> Default : ['loss', 'accuracy'] 

                single_batch    : calculate single batch values.
                    >>> type    : bool
                    >>> Default : True
            
            If model is single_batch, it means that only calculate one batch parameters.
            If model is not single_batch model calculate parameters upto that time. 

            Example of difference between single_batch=True and single_batch=False is loss. 
            When single_batch=True, loss will be equal to that batch's loss. Yet, when single_batch=False,
            loss will be equal to average loss upto that time. 

            To understand when epoch changes, new_epoch function should be called 
            for each epoch, not each batch.
        '''
        self.TRAIN_ONE_BATCH_FLAG = True     

        self._printing_one_batch = printing
        self._val_x = val_x
        self._val_y = val_y
        # if validation rate is not None, split that much data as validation.
        if val_rate:
            assert val_rate != 1.0, 'Validation rate should be less than 1.0 and greater equat than 0.'
            # check whether val_x and val_y is given with val_rate at the same time. If it is given, warn the user.
            assert self._val_x is None and self._val_y is None, 'Validation rate and validation inputs are not used at the same time. '
            # find how many data for validation
            val_ind = int(x_batch.shape[0] * val_rate)
            # shuffle indexing of data to prepare validation then shuffle
            shuff_ind = np.arange(0, x_batch.shape[0])
            np.random.shuffle(shuff_ind)
            x_batch = x_batch[shuff_ind]
            y_batch = y_batch[shuff_ind]
            # split data to validation and trainig.
            self._val_x = x_batch[0:val_ind]
            self._val_y = y_batch[0:val_ind]
            x_batch = x_batch[val_ind:]
            y_batch = y_batch[val_ind:]  

        # make zero of all grads of model because of AD structure.
        self._model.zero_grad()

        # get batch of train and label data                          
        _x_batch = T.make_tensor(x_batch)
        _y_batch = T.make_tensor(y_batch)
        # predict the batch
        _pred = self._feedForward(_x_batch, True)
        # calculate loss, grad, epoch loss which is total loss of epoch and accuracy
        _loss = self._loss.loss(_y_batch, _pred, self._model.get_params()) + self._regularizer()
        _loss.backward()         

        # if model is single_batch, it means that only calculate batch parameters.
        # if model is not single_batch model calculate average loss. To understand when epoch changes, 
        # new_epoch function should be called 
        # for each epoch, not each batch.
        if single_batch:
            self._ite = 1
            self._acc.reset()  
            self._epoch_loss = np.mean(_loss.value)
        else:
            self._ite += 1
            self._epoch_loss += np.mean(_loss.value)

        self._accVal = self._acc.accuracy(_y_batch,_pred)
        # append history of loss and accuracy
        self.his_loss.append(np.round(self._epoch_loss / self._ite, 4))
        self.his_acc.append(np.round(self._accVal, 4))
        # optimize layer parameters
        self._optimizer.step(self._layer)
        # print properties
        self._train_print(printing_var=printing)


    def new_epoch(self):
        """
            Resetting some values of Network. 

            When train_one_batch function used for more than one epoch, this function
            should be called because some parameters should be resetted.
        """
        
        # check where called new_epoch function. It should be at the end of loop
        assert self._printing_one_batch, 'Please use new_epoch at the end of loop.'
        
        # validation same as traning, difference is not calling backpropagation and optimizer
        if self._val_x is not None and self._val_y is not None:
            self.VALIDATION_FLAG = True
            _val_x_batch = T.make_tensor(self._val_x)
            _val_y_batch = T.make_tensor(self._val_y)                    
            _val_pred = self._feedForward(_val_x_batch)
            _val_loss = self._loss.loss(_val_y_batch, _val_pred, self._model.get_params())
            self._val_epoch_loss +=  np.mean(_val_loss.value)
            self._val_accVal = self._val_acc.accuracy(_val_y_batch,_val_pred)
            self._val_ite += 1
        
        # print properties
        self._train_print(end_epoch=False, printing_var=self._printing_one_batch)

        self._epoch_loss = 0
        self._ite = 0
        self._val_epoch_loss = 0
        self._val_ite = 0
        self._acc.reset()
        self._val_acc.reset()


    def setup(self, loss_function = 'categoricalcrossentropy', optimizer = 'adam'):
        '''
            Setup Neural Network by selecting loss function and optimizer.

            Arguments:

                loss_function : loss function of model.

                optimizer     : optimizer of model.

            User have 2 way to select loss function and optimizer.
            First way is select by strings (which can be learn from proper modules or docs). 
            Ex :
                >>> net.setup(loss_function='cce', optimizer='adam')

            Last way is custom loss function or optimizer or both of them.
            Ex : 
                >>> opt = gNet.optimizer.Adam(lr=0.0001)
                >>> net.setup(loss_function='cce',optimizer=opt)
        '''
        # get proper loss function. If argument is string, select it from _lossFuncCaller 
        # dict. If argument is declared explicitly it return to _loss directly.
        if isinstance(loss_function, str):
            _l = loss_function.lower()
            self._loss = self._lossFuncCaller[_l]()
        else:
            self._loss = loss_function        
        # get proper metric from loss function class
        self._acc = self._loss.get_metric() # accuracy metric
        self._val_acc = self._loss.get_metric() # validation metric
        # get proper optimizer. If argument is string, select it forn _optimizerCaller
        # dict. If argument is declared explicitly it return to _optimizer directly.
        if isinstance(optimizer, str):
            _opt = optimizer.lower()
            self._optimizer = self._optimizerCaller[_opt]()
        else:
            self._optimizer = optimizer


    def predict(self, x):
        '''
            Prediction of Neural Network.
            Return depends on model.

            Argument : 

                x   : prediction input.
        '''
        return self._feedForward(T.make_tensor(x))


    def evaluate(self, eva_x=None, eva_y=None):
        '''
            Evaluate function of Neural Network structure. To evalute NN, this function should
            be called. 

            evaluate method can return eva_loss and eva_acc respectively optionally.

            Arguments:
            ----------
                
                eva_x : evaluate data.

                eva_y : label data of evaluation.

        '''
        # create start time to calculate total time for evaluate.
        s_time = time.time()

        # make data to tensor 
        _eva_x = T.Tensor(eva_x)
        _eva_y = T.Tensor(eva_y)

        # get start index of test data w.r.t batch size
        _starts = np.arange(0, _eva_x.shape[0], self._batchSize)

        # initialize evaluate loss and iteration
        _eva_loss = 0
        _ite = 0
        
        # reset accuracy of model to calculate evaluate accuracy.
        self._acc.reset()

        # run batch of evaluate
        for _start in _starts:
            # get last index of batch 
            _end = _start + self._batchSize
            # get batch of data
            _x_batch = _eva_x[_start:_end]
            _y_batch = _eva_y[_start:_end]
            # predict the batch 
            _pred = self._feedForward(_x_batch)
            # calculate accuracy of prediction
            _accValTest = self._acc.accuracy(_y_batch,_pred)
            # calculate loss of prediction to add evaluate loss 
            _loss = self._loss.loss(_y_batch, _pred, self._model.get_params())
            _eva_loss += np.mean(_loss.value)
            _ite += 1

        # print loss, accuracy and passed time for evaluate
        print('Test Loss : {}, Accuracy : {}'.format(np.round(_eva_loss / _ite, 4), np.round(_accValTest, 4)))
        print("Passed Evaluate Time : ", datetime.timedelta(seconds=(time.time()-s_time)))
        return np.round(_eva_loss / _ite, 4), np.round(_accValTest, 4)


    def evaluate_one_batch(self, eva_x_batch=None, eva_y_batch=None, single_batch=True):
        '''
            Evaluate one batch of data.

            Arguments:
            ----------
                
                eva_x           : evaluate data.

                eva_y           : label data of evaluate.

                single_batch    : evaluate single batch values.
                    >>> type    : bool
                    >>> Default : True            

            If model is single_batch, it means that only evaluate one batch parameters.
            If model is not single_batch model evaluate parameters upto that time. 

            Example of difference between single_batch=True and single_batch=False is loss. 
            When single_batch=True, loss will be equal to that batch's loss. Yet, when single_batch=False,
            loss will be equal to average loss upto that time. 
        '''
        # make data to tensor 
        _eva_x = T.Tensor(eva_x_batch)
        _eva_y = T.Tensor(eva_y_batch)
        
        # predict the batch 
        _pred = self._feedForward(_eva_x)
        # calculate accuracy of prediction
        _accValTest = self._acc.accuracy(_eva_y,_pred)
        # calculate loss of prediction to add evaluate loss 
        _loss = self._loss.loss(_eva_y, _pred, self._model.get_params())
        
        # if model is single_batch, it means that only evaluate batch parameters.
        # if model is not single_batch model evaluate average loss. 
        if single_batch:
            self._acc.reset()  
            self._eva_ite = 1
            self._eva_loss = np.mean(_loss.value)
        else:
            # resetting accuracy when evaluate_one_batch called first time
            if self._eva_acc_reset == False:
                self._acc.reset()
                self._eva_acc_reset = True
            self._eva_ite += 1
            self._eva_loss += np.mean(_loss.value)

        # print loss, accuracy and passed time for evaluate
        print(' Loss : {}, Accuracy : {}'.format(np.round(self._eva_loss / self._eva_ite, 4), np.round(_accValTest, 4)), end='\r')


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
        sm = []
        # added each layer's trainable parameters to list 
        for layer in self._layer:
            app_item = layer.trainable
            # if layer is Batch Norm. save also running mean and running variance
            if self._model.get_params()['layer_name'][layer._thisLayer] == 'Batch Normalization':
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
        model_layer_name = self._model.get_params()['layer_name']
        fName = file_name + '.npy'
        w = np.load(fName, allow_pickle=True)
        # check layer properties is same as saved ones.
        for ind, layer in enumerate(self._layer):
            for ind_tra, trainable in enumerate(layer.trainable):
                # if layer is Batch Norm. load also running mean and running variance
                if model_layer_name[ind] == 'Batch Normalization':
                    tmp = w[ind][ind_tra]
                    t_shape = tmp.shape
                    layer._r_mean = w[ind][2]
                    layer._r_var = w[ind][3]
                else:
                    t_shape = w[ind][ind_tra].shape

                assert trainable.shape == t_shape, \
                    str('Check ' + model_layer_name[ind] + ' or Layer No:'+str(ind) \
                        +' parameters of model. \n'\
                        'Loaded model are not proper.\n' + \
                        'Model :' + str(trainable.shape) + \
                        '\tLoaded :' +str(t_shape))
            layer.trainable = w[ind]
        # if everythings passed, print the success.
        if os.path.isfile(fName):
            print('Model weights of `' + fName + '` loaded successfully..')


    def get_loss_plot(self, show=False, save=False, figure_name='gNet_loss.png', 
                    figure_title='Loss vs Iterations', x_label='Iterations', y_label='Loss'):
        '''
            Get loss plot of training of Neural Network. Plot can be showed, saved or both of them.

            Arguments:
            ---------
                
                show                : show the figure of loss during training.
                    >>> type        : bool
                    >>> Default     : False

                save                : save the figure of loss during training.
                    >>> type        : bool
                    >>> Default     : False
                     
                figure_name         : name of file which store the plot of loss.
                    >>> type        : string
                    >>> Default     : gNet_loss.png 

                figure_title        : set title of plot.
                    >>> type        : string
                    >>> Default     : Loss vs Iterations

                x_label             : set x_label of plot.
                    >>> type        : string
                    >>> Default     : Iterations

                y_label             : set y_label of plot.
                    >>> type        : string
                    >>> Default     : Loss
        '''
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.his_loss)
        ax.set_title(figure_title)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.grid()

        if save:
            fig.savefig(figure_name)
        
        if show:
            plt.show()


    def get_accuracy_plot(self, show=False, save=False, figure_name='gNet_accuracy.png', 
                        figure_title='Accuracy vs Iterations', x_label='Iterations', y_label='Accuracy'):
        '''
            Get accuracy plot of training of Neural Network. Plot can be showed, saved or both of them.

            Arguments:
            ---------
                
                show                : show the figure of loss during training.
                    >>> type        : bool
                    >>> Default     : False

                save                : save the figure of loss during training.
                    >>> type        : bool
                    >>> Default     : False
                     
                figure_name         : name of file which store the plot of accuracy.
                    >>> type        : string
                    >>> Default     : gNet_accuracy.png 

                figure_title        : set title of plot.
                    >>> type        : string
                    >>> Default     : Accuracy vs Iterations

                x_label             : set x_label of plot.
                    >>> type        : string
                    >>> Default     : Iterations

                y_label             : set y_label of plot.
                    >>> type        : string
                    >>> Default     : Accuracy
        '''
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.his_acc)
        ax.set_title(figure_title)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.grid()

        if save:
            fig.savefig(figure_name)
        
        if show:
            plt.show()


    def get_model_summary(self, show=True, save=False, summary_name='gNet_model_summary.txt'):
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
        '''
        # get model parameters
        model_act = self._model.get_params()['layer_name']
        output_shape = self._model.get_params()['layer_output_shape']
        params_no = self._model.get_params()['#parameters']   
        # create texttable
        t = Texttable()
        t.add_rows([['Layer', 'Output Shape', '# of Parameters']])
        for ind in range(len(model_act)):
            tmp = [str(ind)+ ': '+model_act[ind], output_shape[ind], params_no[ind]]
            t.add_row(tmp)
        t.add_row(['Total', ' ', '{:,}'.format(np.sum(params_no))])
        if show:
            print(t.draw())
        if save:
            f = open(summary_name, 'w')
            f.write(t.draw())

