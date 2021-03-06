# gNet

[![GitHub release](https://img.shields.io/github/v/release/MGokcayK/gNet)](https://github.com/MGokcayK/gNet/releases/)
[![PyPI version shields.io](https://img.shields.io/pypi/v/gNet)](https://pypi.python.org/pypi/gNet/)
[![PyPI license](https://img.shields.io/pypi/l/gNet.svg)](https://pypi.python.org/pypi/gNet/)
[![Docs](https://img.shields.io/badge/Docs-readible-green.svg)](https://github.com/MGokcayK/gNet/blob/master/docs/)

gNet is a mini Deep Learning(DL) library. It is written to understand how DL
works. It is running on CPU. It is written on Python language and used :
    
    * Numpy for linear algebra calculations
    * Matplotlib for plottings
    * Texttable for proper printing of model summary in cmd
    * wget for download MNIST data
    * idx2numpy for load MNIST data
    
some 3rd party libraries.

During devolopment, Tensorflow, Keras, Pytorch and some other libraries examined.
Keras end-user approach is used. Therefore, if you are familiar with Keras,
you can use gNet easily.

gNet has not a lot functions and methods for now, because subject is written when
they needed to learn. Also, gNet is personal project. Thus, its development process
depends on author learning process.

## Installation

Installation can be done with pip or clone the git and use in local file of your workspace.

To install with [pip](https://pypi.org).

```bash
pip install gNet
```

## Example - MNIST

### Sequential Model
```python

from gNet import utils
from gNet import neuralnetwork as NN
from gNet import model
from gNet import layer
from gNet import optimizer
from gNet import loss_functions as LF

# download and load MNIST Dataset
mnist = utils.MNIST_Downloader()
x_train, y_train = mnist.load_train()
x_test, y_test = mnist.load_test()

# normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# make one-hot vector to label
num_classes = 10
y_train = utils.make_one_hot(y_train, num_classes)
y_test = utils.make_one_hot(y_test, num_classes)

# create model
model = model.Model()

# add layers 
model.add(layer.Flatten(input_shape=x_train[0].shape))
model.add(layer.Dense(128, 'relu'))
model.add(layer.Dense(10, 'softmax'))

# create NN structure
net = NN.NeuralNetwork(model)

# print model summary firstly
net.get_model_summary()

# setup structure
net.setup(loss_function='cce', optimizer='adam')

# train 
net.train(x_train, y_train, batch_size=32, epochs=10)

# evaluate
net.evaluate(x_test, y_test)

# get loss and accuracy plot
net.get_loss_plot(show=True)
net.get_accuracy_plot(show=True)

```

Result will be like :
```
Model created and initializing parameters..

+--------------------+--------------+-----------------+
|       Layer        | Output Shape | # of Parameters |
+====================+==============+=================+
| 0: flatten         | 784          | 0               |
+--------------------+--------------+-----------------+
| 1: Dense : relu    | 128          | 100480          |
+--------------------+--------------+-----------------+
| 2: Dense : softmax | 10           | 1290            |
+--------------------+--------------+-----------------+
| Total              |              | 101,770         |
+--------------------+--------------+-----------------+

Train starting..

Epoch : 1 / 10   100.00 %  Loss : 0.2640  Accuracy : 0.9241
Epoch : 2 / 10   100.00 %  Loss : 0.1164  Accuracy : 0.9657
Epoch : 3 / 10   100.00 %  Loss : 0.0802  Accuracy : 0.9761
Epoch : 4 / 10   100.00 %  Loss : 0.0598  Accuracy : 0.9816
Epoch : 5 / 10   100.00 %  Loss : 0.0469  Accuracy : 0.9856
Epoch : 6 / 10   100.00 %  Loss : 0.0373  Accuracy : 0.9884
Epoch : 7 / 10   100.00 %  Loss : 0.0301  Accuracy : 0.9908
Epoch : 8 / 10   100.00 %  Loss : 0.0234  Accuracy : 0.9931
Epoch : 9 / 10   100.00 %  Loss : 0.0213  Accuracy : 0.9933
Epoch : 10 / 10   100.00 %  Loss : 0.0164  Accuracy : 0.9949
Passed Training Time :  0:01:04.485637
Test Loss : 0.0969, Accuracy : 0.9747
Passed Evaluate Time :  0:00:00.140604
```
### Functional Connection Layer Model

```python

class MnistTrainer():
    def __init__(self) -> None:
        self.batchSize = 32
        self.epoch = 10
        self.createModel()
        self.loss = LF.CategoricalCrossEntropy()
        self.acc = self.loss.get_metric()
        self.layers = self.output.get_layers() # get all connectec layer from input layer.
        self._optimizer = optimizer.Adam()
        self.output.get_model_summary() # get model summary

    def createModel(self):
        self.flatten = layer.Flatten(input_shape=x_train[0].shape)
        self.flatten() # calculate layer properties as input layer.
        self.h1 = layer.Dense(128,'relu')
        self.h1(self.flatten) # connect the hidden layer to flatten layer as previous layer.
        self.output = layer.Dense(10, 'softmax')
        self.output(self.h1)
    
    # compute model layer by layer
    def compute(self, inputs, train=True):
        x = self.flatten.compute(inputs, train)
        x = self.h1.compute(x, train)
        return self.output.compute(x, train)

    def train(self):
        for e in range(self.epoch):
            self._ite = 0
            self.acc.reset()
            self._starts = np.arange(0, x_train.shape[0], self.batchSize)
            self._epoch_loss = 0
            for _start in self._starts:
                self._ite += 1
                _end = _start + self.batchSize
                _x_batch = T.make_tensor(x_train[_start:_end])
                _y_batch = T.make_tensor(y_train[_start:_end])

                self.output.zero_grad() # zeroing all layers' grad by calling `zero_grad`

                _pred = self.compute(_x_batch, True)
                _loss = self.loss.loss(_y_batch, _pred, self.output)
                _loss.backward()
                self._epoch_loss += np.mean(_loss.value)           
                self._accVal = self.acc.accuracy(_y_batch,_pred)    

                self._optimizer.step(self.layers)

                printing = 'Epoch : %d / %d ' % (e + 1, self.epoch)
                printing += ' Loss : %.4f ' % (np.round(self._epoch_loss / self._ite, 4))
                printing += ' Accuracy : %.4f ' % (np.round(self._accVal, 4))
                print(printing, end='\r')
            print("")

net = MnistTrainer()
net.train()
```

Result will be like :
```
Model created and initializing parameters..
+-----------------------------------+--------------+-----------------+
| Layer No (Previous Layer) | Layer | Output Shape | # of Parameters |
+===================================+==============+=================+
| 0: flatten                        | 784          | 0               |
+-----------------------------------+--------------+-----------------+
| 1(0) | Dense : relu               | 128          | 100480          |
+-----------------------------------+--------------+-----------------+
| 2(1) | Dense : softmax            | 10           | 1290            |
+-----------------------------------+--------------+-----------------+
| Total                             |              | 101,770         |
+-----------------------------------+--------------+-----------------+
Epoch : 1 / 10  Loss : 0.2720  Accuracy : 0.9221
Epoch : 2 / 10  Loss : 0.1200  Accuracy : 0.9649
Epoch : 3 / 10  Loss : 0.0806  Accuracy : 0.9762
Epoch : 4 / 10  Loss : 0.0588  Accuracy : 0.9829
Epoch : 5 / 10  Loss : 0.0442  Accuracy : 0.9875
Epoch : 6 / 10  Loss : 0.0330  Accuracy : 0.9912
Epoch : 7 / 10  Loss : 0.0249  Accuracy : 0.9937
Epoch : 8 / 10  Loss : 0.0197  Accuracy : 0.9950
Epoch : 9 / 10  Loss : 0.0172  Accuracy : 0.9951
Epoch : 10 / 10  Loss : 0.0144  Accuracy : 0.9959
```


## Details

Details can be found in [mini docs](../master/docs/gNet-v0.1.pdf).

## License
[MIT](https://choosealicense.com/licenses/mit/)
