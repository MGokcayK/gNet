"""
    Utility module of gNet.
    
    Author : @MGokcayK 
    Create : 03 / 07 / 2020
    Update : 08 / 07 / 2020
                Added MNIST downloader.
"""

import os 
import gzip
import wget
import idx2numpy
import numpy as np

def make_one_hot(data, size):
    """
        Making one hot vector.
        
        Arguments:
        ----------
            
            data : data which stores indexes.

            size : size of output shape
    """
    shape = (data.size, size)
    one_hot = np.zeros(shape)
    rows = np.arange(data.size)
    one_hot[rows, data] = 1
    return one_hot


class MNIST_Downloader():
    """
        MNIST Downloader of gNet. 


        from gNet import utils 
        ...
        mnist = utils.MNIST_downloader()
        x_train, y_train = mnist.load_train()
        x_test, y_test = mnist.load_test()
        
        #if normalized 
        # x_train, x_test = x_train / 255.0, x_test / 255.0
        ...
    """
    def __init__(self):
        self.working_directory = os.getcwd()
        self.dir_name = 'MNIST'
        
        self.main_url = 'http://yann.lecun.com/exdb/mnist/'
        self.train_file_name = 'train-images-idx3-ubyte.gz'
        self.train_label_file_name = 'train-labels-idx1-ubyte.gz'
        self.test_file_name = 't10k-images-idx3-ubyte.gz'
        self.test_label_file_name = 't10k-labels-idx1-ubyte.gz'

        self._check_directory()
        self._download()
        self._decompress()

    def _check_directory(self):
        # checking whether there is intended directory. It it is not, make it.
        if not os.path.isdir(self.dir_name):
            os.makedirs(self.dir_name)

    def _change_dir(self):
        # make sure that we are in working directory
        os.chdir(self.working_directory)
        # change directory to target directory to do purpose
        os.chdir(self.dir_name)
    
    def _download(self):
        # download mnist data 
        self._download_train()
        self._download_train_label()
        self._download_test()
        self._download_test_label()

    def _download_train(self):
        self._change_dir()
        # download data
        if not os.path.exists(self.train_file_name):
            print('\nTrain file downloading..\n')
            wget.download(self.main_url + self.train_file_name, self.train_file_name)
        # back to working directory
        os.chdir(self.working_directory)

    def _download_train_label(self):
        self._change_dir()
        # download data
        if not os.path.exists(self.train_label_file_name):
            print('\nTrain label file downloading..\n')
            wget.download(self.main_url + self.train_label_file_name, self.train_label_file_name)
        # back to working directory
        os.chdir(self.working_directory)
        
    def _download_test(self):
        self._change_dir()        
        # download data
        if not os.path.exists(self.test_file_name):
            print('\nTest file downloading..\n')
            wget.download(self.main_url + self.test_file_name, self.test_file_name)
        # back to working directory
        os.chdir(self.working_directory)
        
    def _download_test_label(self):
        self._change_dir()
        # download data
        if not os.path.exists(self.test_label_file_name):
            print('\nTest label file downloading..\n')
            wget.download(self.main_url + self.test_label_file_name, self.test_label_file_name)
        # back to working directory
        os.chdir(self.working_directory)        

    def _decompress(self):
        self._change_dir()
        # decompress data
        for item in os.listdir():
            if item[-3:]=='.gz':
                pressed = gzip.GzipFile(item, 'rb')
                stack = pressed.read()
                pressed.close()

                output = open(item[:-3], 'wb')
                output.write(stack)
                output.close()
        # back to working directory
        os.chdir(self.working_directory)    

    def load_train(self):        
        """
            Load train data of MNIST. 

            Return will be trainin data (x) and its label (y) respectively. 
        """
        self._change_dir()
        # decompress data
        x = idx2numpy.convert_from_file(self.train_file_name[:-3])
        y = idx2numpy.convert_from_file(self.train_label_file_name[:-3])
        # back to working directory
        os.chdir(self.working_directory)
        return x, y

    def load_test(self):        
        """
            Load test data of MNIST. 

            Return will be test data (x) and its label (y) respectively. 
        """
        self._change_dir()
        # decompress data
        x = idx2numpy.convert_from_file(self.test_file_name[:-3])
        y = idx2numpy.convert_from_file(self.test_label_file_name[:-3])
        # back to working directory
        os.chdir(self.working_directory)
        return x, y
    