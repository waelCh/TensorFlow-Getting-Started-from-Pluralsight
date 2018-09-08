#
#   This file predictes Houses price using Tensorflow
#
#
#
#

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as  plt
import matplotlib.animation as animation  # import animation support 

#   Generate some house sizes beteen 1000 and 3500
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

#Generate house prices from house size with a random noise added
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

#   Plot generated house and size
plt.plot(house_size, house_price, "bx") #   bx = blue x
plt.ylabel("Price")
plt.xlabel("size")
plt.show()

#   Normalizer function to prevent under/overflows
def normalize(array):
    return (array - array.mean()) / array.std()

#   Define number of training sample, 0.7 = 70%; we can take the first 70% since we are randomized
num_train_samples = math.floor(num_house * 0.7)

#   Define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_house_size)

#   Define test data
test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

test_house_price_norm = normalize(test_house_price)
test_house_size_norm = normalize(test_house_size)

#   Set up the placeholders that get updated as we desceed down the greadient
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

#   Define the variables holding the size_factor and price we set during training
#   We initialize them to some random values based on the normal distribuion 
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

#   Define the operation for the predicting values - predicted price = (size_factor * house_size) + price_offset
#   Notice the use of the Tensorflow add and multiply functon.
#   These add the operation to the tensorflow computation graph
tf_price_pred = tf.add(tf.multiply(tf_size_factor, house_size), tf_price_offset)

#   Define the loss function (how much error) - Mean squared error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price, 2))/(2*num_train_samples)

#   Optimizer learning rate. The size of the steps down the gradient
learning_rate = 0.1

#   4.Define a Gradient descent optimizer that will minimize the loss defined in the operation "cost"
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

#Initialize the variables
init = tf.global_variables_initializer()

#   Launch the graph in the session
with tf.Session as sess:
    sess.run(init)

    #   Set how often to display progress and number of training iterations
    display_every = 2
    num_training_iter = 50

    #   Keep iterating in the training data
    for iteration in range(num_training_iter):
        
        #   Fit all training data
        for (x, y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

        # Display current status
        if (iteration+1)%display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
            print("iteration #:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c), \
                "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))
            
            print("Optimization finished !")
            training_cost = sess.run(tf_cost, feed_dict={tf})

