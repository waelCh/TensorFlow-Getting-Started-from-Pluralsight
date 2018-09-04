#   import Tensorflow
import tensorflow  as tf

sess = tf.Session()

# Verify we can print a string
hello = tf.constant('Assalamou 3alaykom ...')
print(sess.run(hello))

#   Perform some simple Math
a = tf.constant(20)
b = tf.constant(30)
print('a + b = {0}'.format(sess.run(a+b)))