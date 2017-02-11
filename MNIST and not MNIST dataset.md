[TOC]

# MNIST

MNIST is an acronym for "Mixed National Institute of Standards and Technology". The original dataset is hosted by [Yann Lecun](http://yann.lecun.com/exdb/mnist/index.html). As early as 1998, he used 7-layer convolutional net **LeNet-5** to achieve an error rate of 0.8. 

The data is packaged into 4 .gz files, with a total size of 12.5 MB. 

[train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz):  training set images (9912422 bytes) 
[train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz):  training set labels (28881 bytes) 
[t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz):   test set images (1648877 bytes) 
[t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz):   test set labels (4542 bytes)

- 70 k instances: 60 k training + 10 k testing.
- 784 input features, or 28\*28  pixels

The data is written in bytestream format, which is not ready for immediate use. Let's see how different teams preprocess it to suit their needs.

##Preprocessed by TensorFlow

If you follow the tutorial, you will see: 

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
for i in range(1000): # iteration
    batch_xs, batch_ys = mnist.train.next_batch(100) # batch size
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

`mnist.train.next_batch()` seems to be a black box that does the magic.  However, `input_data.py` does nothing but throws the dirty work to `mnist.py` which has a function `read_data_set()`: 

```python
# tensorflow/examples/tutorials/input_data.py
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# tensorflow/contrib/learn/python/learn/datasets/mnist.py
def read_data_sets(train_dir, fake_data=False, one_hot=False,dtype=dtypes.float32,reshape=True,validation_size=5000):
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    local_file = base.maybe_download(TRAIN_IMAGES, train_dir, SOURCE_URL + TRAIN_IMAGES)
    with open(local_file, 'rb') as f:
    train_images = extract_images(f)  # dirty job here
    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    return base.Datasets(train=train, validation=validation, test=test)
    
def extract_images(f):
    def _read32(bytestream):
    	dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    	return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
    return data
```

We can cleary see the source is Yann Lecun. The source file is a GzipFile with a special data type `numpy.dtype(numpy.uint32).newbyteorder('>')` and `numpy.uint8`. The reading command is `numpy.frombuffer(buf, dtype)`. The data is first reshaped to a 4D array of num\*28\*28\*1, then reshape to a 2D array of num\*784. 

In `Dataset.next_batch(batch_size)`, what it does is use `_index_in_epoch` to track the already trained example. when it overpass the total examples, shuffle the examples and start over from 0. 

## Preprocessed by Michael Nielsen 

When  I read his amazing book "neural network and deep learning", I was tring to understand his elegant [code](https://github.com/mnielsen/neural-networks-and-deep-learning/tree/master/data):

```python
# mnist_loader.load_data()
with gzip.open('mnist.pkl.gz', 'rb') as f:
    training_data, validation_data, test_data = cPickle.load(f)
[X_train, y_train] = training_data
```

The daa is already well-organized in this mnist.pkl.gz.  The loaded file is a 3-element tuple (training_data, validation_data, test_data). Each data is a 3D array. The 1st dimension is  [X,y], the 2nd dimension is the number of instances, and the 3rd dimension are features (784 features for X, 1 feature for y). As a result, X_train,y_train can be directly used to fit `sklearn.svm.SVC()`. This is so handy.

### Nielsen's code network1

Nielsen's approach to feed the data into a manual SGD network is to **pair** each single training features and label as  a tuple by `zip()`. A trick has to be played: each (1,n) shape X_train is converted to (n,1) shape by `np.reshape`

```python
    training_inputs = [np.reshape(x, (num, 1)) for x in X_train]  # 1* n to n*1
    training_results = [np.reshape(y,(1,1)) for y in y_train]
    training_data = zip(training_inputs, training_results)
```

### Nielsen's code network3

But in  convolutional nets, Nielsen had to give up his manual coding approach and used **theano API**, which requires to put X,y data  into "shared" placeholder individually. 

```python
def shared(X_train,y_train):
        shared_x = theano.shared(
           np.asarray(X_train, dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
           np.asarray(y_train, dtype=theano.config.floatX), borrow=True)
        return (shared_x, shared_y) 
```

And the dirty job is done by `theano.function()`

```python
i = theano.tensor.lscalar() # mini-batch index
train_mb = theano.function([i], cost, updates=updates, givens={self.x: training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
        self.y:training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]})
```

## sklearn

```python
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")
X = mnist.data / 255. # shape (70000,784)
y = mnist.target      # shape (70000,)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
```

And it's very satisfying to see an test accuracy of 97.1% at your finger tips.

```python
from time import time
t0 = time()
if False: # final loss = 0.016, test score = 97.58%, time = 387 s
  mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
if True: # final loss = 0.047, test score = 97.10%, time = 7.7s
  mlp = MLPClassifier(hidden_layer_sizes=(50,) ,max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,learning_rate_init=.1)
mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))
print("Time cost: {0}".format(time()-t0))
```

Note that the default activation is '**relu**'.  This is almost the best job sklearn can do for you. **No convolutional layers here**. Thank you very much. 

# notMNIST

Hosted by Yaroslav Bulatov: http://yaroslavvb.com/upload/notMNIST/

or an internal url of google: http://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz

"notMNIST_Large.tar.gz" is ~250 MB, the "notMNIST_Small.tar.gz" is ~8 MB, which serves as test set.

- 519 k instances: 500 k training+ 29 k testing
- 519 k png files, each is 28\*28 pixels
- 10-class labels from 'A' to "J",  

use **tarfile.open(), extractall()** to open the .tar.gz file, extract to 10 file folders, each folder is a collection of the letter writing, e.g. 'A', with 52 k png files. The folder names are stored in variable called "**train_folders**" or "test_folders",  a list of strings.

use **scipy.ndimage.imread()** to convert a png file into a 2D imbeded array (28\*28), normalize values to (-0.5,0.5) by dividing the pixel depth 255.  AS a result, we get a 52.9 k\*28\*28 numpy array.

