import tensorflow as tf

x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5], [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])
W = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Correct prediction Test model
prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph
with tf.Session() as sess:
   # Initialize TensorFlow variables
   sess.run(tf.global_variables_initializer())
   for step in range(201):
       cost_val, W_val, _ = sess.run([cost, W, optimizer], 
                       feed_dict={X: x_data, Y: y_data})
       print(step, cost_val, W_val)

   # predict
   print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
   # Calculate the accuracy
   print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))



import numpy as np
from sklearn.preprocessing import MinMaxScaler

xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
              [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
              [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
              [816, 820.958984, 1008100, 815.48999, 819.23999],
              [819.359985, 823, 1188100, 818.469971, 818.97998],
              [819, 823, 1198100, 816, 820.450012],
              [811.700012, 815.25, 1098100, 809.780029, 813.669983],
              [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

minMaxScaler = MinMaxScaler()
xy = minMaxScaler.fit_transform(xy)
print(xy)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
   cost_val, hy_val, _ = sess.run(
       [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
   print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)



from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_xs, batch_ys = mnist.train.next_batch(100)

nb_classes = 10

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
   # Initialize TensorFlow variables
   sess.run(tf.global_variables_initializer())
   # Training cycle
   for epoch in range(training_epochs):
       avg_cost = 0
       total_batch = int(mnist.train.num_examples / batch_size)

       for i in range(total_batch):
           batch_xs, batch_ys = mnist.train.next_batch(batch_size)
           c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
           avg_cost += c / total_batch

       print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
   print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
   r = random.randint(0, mnist.test.num_examples - 1)
   print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
   print("Prediction:",sess.run(tf.argmax(hypothesis, 1),feed_dict={X: mnist.test.images[r:r + 1]}))   # Get one and predict


plt.imshow(mnist.test.images[r:r + 1].
            reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()



#iris data
import tensorflow as tf
import numpy as np
from numpy import array
from sklearn.datasets import load_iris
import random

iris_dataset = load_iris()

x = iris_dataset['data']
y = iris_dataset['target']

x = np.float32(x)

#y = array(y)
y = np.reshape(y, (150,1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

nb_classes = 3

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.int32, [None, 1])

y_one_hot = tf.one_hot(Y, nb_classes)
y_one_hot = tf.reshape(y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([4, nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

# Hypothesis (using softmax)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(y_one_hot, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
training_epochs = 15
batch_size = 35

with tf.Session() as sess:
   # Initialize TensorFlow variables
   sess.run(tf.global_variables_initializer())
   # Training cycle
   for epoch in range(training_epochs):
       avg_cost = 0
       total_batch = int(len(X_train) / batch_size)

       for i in range(total_batch):
           batch_xs, batch_ys = X_train[i*batch_size:i*batch_size+batch_size], y_train[i*batch_size:i*batch_size+batch_size]
           c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
           avg_cost += c / total_batch

       print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
   print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: X_test, Y: y_test}))
   r = random.randint(0, len(X_test) - 1)
   print("Label:", sess.run(tf.argmax(y_test[r:r+1], 1)))
   print("Prediction:",sess.run(tf.argmax(hypothesis, 1),feed_dict={X: X_test[r:r + 1]}))
