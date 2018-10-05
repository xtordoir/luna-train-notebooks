<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>

# Tensorflow Part III: More on Linear Regression 

## Goals

- Work with Training/Validation set
- Compute Loss, RMSE
- Placeholders
- Tensorflow DataSet
- Estimators

## 3. Validation



We start from the previous implementation:

```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

```
fileurl = "https://s3-eu-west-1.amazonaws.com/training180529/data/djia_close.csv"
df = pd.read_csv(fileurl, sep=',',header=0)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by= 'date').reset_index(drop=True)
```

```
learning_rate = 1e-8
training_steps = 10000
losses = []

with tf.Session() as sess:
    x = tf.constant(df.drop(columns=['date', 'JPM', 'DWDP', 'MMM']))
    #x = tf.constant(df[['AXP']])
    y = tf.constant(df[['JPM']])
    
    weights = tf.Variable(tf.random_normal([27, 1], 0, .1, dtype=tf.float64))
    
    b = tf.Variable(tf.random_normal([1], 0, 0.1, dtype=tf.float64))
    
    tf.global_variables_initializer().run()
    yhat = x @ weights + b
    yerror = tf.subtract(yhat, y)
    
    loss = tf.nn.l2_loss(yerror)
    update_weights = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    for _ in range(training_steps):
        # Repeatedly run the operations, updating the TensorFlow variable.
        sess.run(update_weights)
        losses.append(loss.eval())
        #print(gradient.eval())
    betas = weights.eval()
    bias = b.eval()
    yhat = yhat.eval()
    y = y.eval()

```

```
print(bias)
print(betas.reshape(27))
```

```
plt.yscale('log')
plt.plot(range(0, training_steps), losses)
```

![alt text](img/losses.png "losses")

### Training/Validation sets

We see the optimiser reduces the Loss which is computed on the training data. We would like to know the quality of the model on new data. For this we must train on part of the data (e.g. 2/3) and measure model quality on the remaining data.

We start by splitting the dataframe `df``as a training set and test set.

```
train=df.sample(frac=0.66)
test=df.drop(train.index)
```

Then we create the corresponding tensorflow variables:

x_train, y_train

and

x_test, y_test

#### Exercise 

Run a the training minimising the training set loss. Compute the Root Mean Squared Error on the test and training set.

$$
RMSE = \sqrt{\frac{1}{n}\sum{(y-\hat{y})^2}}
$$

Tips:

- RMSE is the square root of the mean squared error, and sum of squared errors is the Loss as calculated before (L2).

- Square root function from `math` module:

```
import math
math.sqrt(...)
```

Compare RMSE for training and test set, why the difference? Which validation set is the most relevant?


#### Exercise 

Use as training set the first 2/3 of the dates (no random pick), and test set is the last 1/3 of the time series.

Again, compare the RMSE and explain the difference.


e.g.

-

$$
\begin{matrix}
Set & RMSE \\\
\hline
Training (Random) & 0.83 \\\
Validation (Random) & 0.93 \\\
Validation (Sequence) & 1.50
\end{matrix}
$$

-

#### Exercise

There is a `tf.metrics.mean_squared_error` function:

https://www.tensorflow.org/api_docs/python/tf/metrics/mean_squared_error

Modify the notebook to use this function to report RMSE.

## 4. TensorFlow Dataset

We will now consume data in an iterative fashion, using the Dataset API of tensorflow.

### !!! Save a csv file !!!

First, we need to save the pandas dataframe as csv, for exampe in `/tmp/djia.csv`, because tensorflow will not read http directly. We will clean, removing unused columns (`date`, `DWDP`, `MMM`):

```
fileurl = "https://s3-eu-west-1.amazonaws.com/training180529/data/djia_close.csv"
df = pd.read_csv(fileurl, sep=',',header=0)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by= 'date').reset_index(drop=True)
df = df.drop(columns=['date', 'DWDP', 'MMM'])
df.to_csv("/tmp/djia.csv", index=False)
```

### Create a file queue and TextLineReader

From a list of strings (file paths to process), we can create a queue to iterate on:

```
filename_queue = tf.train.string_input_producer(["/tmp/djia.csv"])
```
The `tf.TextLineReader` class is a Reader for text files, on line at a time:

```
reader = tf.TextLineReader(skip_header_lines = True)
```

A `read` operation will return 2 elements: the filename and the line as sting:

```
key, value = reader.read(filename_queue)
```

In the Tensorflow session, we can feed the queue, and before ending the session, we need to close the queue:


```
with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # do something here
    
    coord.request_stop()
```

#### Exercise

In the session, run one read to print `key` and `value` 

### Decode the csv, construct feature/label
`value`can be decoded as csv, using `tf.decode_csv`. It returns a list of tensors, and requires `record_defaults` as a python list for default values and types.

For example, the following csv line

```
1.23,2.33,3.23
```

can be decoded with:

```
decoded = tf.decode_csv(value, record_defaults = [[0.0], [0.0], [0.0]])
```

Note the `record_defaults` structure.

#### Exercise

How to write a decoded for the djia.csv file?
Hints to create `record_defaults`:
- `np.full()` function
- `ndarray.tolist` numpy function

#### Exercise

Write instruction to create `label` as the tensor containing `JPM` value (note that we reference columns with index, not name)

Write instruction to create a `feature` as the tensor containing all but `JPM` values (use `tf.stack`).

### Batch to update 

The training procedure consists in updating parameters from data. Each iteration is using a (mini-)batch of data:

```
  example_batch, label_batch = tf.train.shuffle_batch(
      [features, label], batch_size = 100, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
      
```

Loop on 10 iterations, and check there are updated batches!

## 5. Train the Linear Model in mini-batches...

Left as exercise:

- Train the model using these mini-batches (100 samples)

## 6. Estimator based implementation

Left as exercise:

Estimators provide a LinearModel implementation:

https://www.tensorflow.org/api_docs/python/tf/estimator/LinearRegressor
