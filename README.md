# gNet

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


## Example - MNIST

```python

from gNet import utils
from gNet import neuralnetwork as NN
from gNet import model
from gNet import layer

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

Model created and initializing parameters..

```
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
## Details

Details can be found in [mini docs](../master/docs/gNet-v0.1.pdf).

## License
[MIT](https://choosealicense.com/licenses/mit/)
