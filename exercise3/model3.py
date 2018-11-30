import tensorflow as tf
import numpy as np

class Model:

    def __init__(self, num_filters, filter_size, lr):

        self.x_image = tf.placeholder(tf.float32, shape=[None,96,96,1], name='x')
        self.y_ = tf.placeholder(tf.float32, shape=[None, 5], name='y_')
        self.y_conv = tf.placeholder(tf.float32, shape=[None, 5], name='y_conv')

        self.W_conv1 = self.weight_variable([filter_size, filter_size, 1, num_filters])
        self.b_conv1 = self.bias_variable([num_filters])

        self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)

        self.W_conv2 = self.weight_variable([filter_size, filter_size, num_filters, num_filters])
        self.b_conv2 = self.bias_variable([num_filters])

        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self.max_pool_2x2(self.h_conv2)

        self.W_conv3 = self.weight_variable([filter_size, filter_size, num_filters, num_filters])
        self.b_conv3 = self.bias_variable([num_filters])

        self.h_conv3 = tf.nn.relu(self.conv2d(self.h_pool2, self.W_conv3) + self.b_conv3)
        self.h_pool3 = self.max_pool_2x2(self.h_conv3)

        self.W_conv4 = self.weight_variable([filter_size, filter_size, num_filters, num_filters])
        self.b_conv4 = self.bias_variable([num_filters])

        self.h_conv4 = tf.nn.relu(self.conv2d(self.h_pool3, self.W_conv4) + self.b_conv4)
        self.h_pool4 = self.max_pool_2x2(self.h_conv4)

        self.W_fc1 = self.weight_variable([6 * 6 * num_filters, 400])
        self.b_fc1 = self.bias_variable([400])

        self.h_pool4_flat = tf.reshape(self.h_pool4, [-1, 6*6*num_filters])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool4_flat, self.W_fc1) + self.b_fc1)

        self.W_fc2 = self.weight_variable([400, 5])
        self.b_fc2 = self.bias_variable([5])

        self.y_conv=tf.nn.softmax(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)

        self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y_conv))
        self.train_step = tf.train.GradientDescentOptimizer(lr).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))

        self.prediction = tf.argmax(self.y_conv,1, name='prediction')
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accuracy')

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)
        return file_name

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.00000001)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.00000001, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
