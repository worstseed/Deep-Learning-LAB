import tensorflow as tf
import os

class Model:
    def __init__(self, history_length, batch_size, set_to_default):

        self.set_parameters(set_to_default=set_to_default)

        self.x = tf.placeholder(tf.float32,
                               shape=[None, self.img_shape[0], self.img_shape[1], history_length],
                               name='x')

        y_label = tf.placeholder(dtype=tf.float32,
                                     shape = [None, self.num_classes],
                                     name = "y_label")

        self.y_true = tf.placeholder(tf.float32,
                                     shape=[None, self.num_classes],
                                     name='y_true')

        y_true_cls = tf.argmax(self.y_true, axis=1)

        layer_conv1, self.weights_conv1 = self.new_conv_layer(input=self.x,
				                                             num_input_channels=history_length,
				                                             filter_size=self.filter_size1,
				                                             num_filters=self.num_filters1,
				                                             use_pooling=True)

        layer_conv2, self.weights_conv2 = self.new_conv_layer(input=layer_conv1,
				                                             num_input_channels=self.num_filters1,
				                                             filter_size=self.filter_size2,
				                                             num_filters=self.num_filters2,
				                                             use_pooling=True)

        layer_conv3, self.weights_conv3 = self.new_conv_layer(input=layer_conv2,
				                                             num_input_channels=self.num_filters2,
				                                             filter_size=self.filter_size3,
				                                             num_filters=self.num_filters3,
				                                             use_pooling=True)

        layer_conv4, self.weights_conv4 = self.new_conv_layer(input=layer_conv3,
				                                             num_input_channels=self.num_filters3,
				                                             filter_size=self.filter_size4,
				                                             num_filters=self.num_filters4,
				                                             use_pooling=True)

        layer_flat, num_features = self.flatten_layer(layer_conv4)


        layer_fc1 = self.new_fc_layer(input=layer_flat,
                         			 num_inputs=num_features,
                         			 num_outputs=self.fc_size,
                         			 use_relu=True)

        layer_fc2 = self.new_fc_layer(input=layer_fc1,
                         			 num_inputs=self.fc_size,
                         			 num_outputs=self.num_classes,
                         			 use_relu=False)

        self.y_pred = tf.nn.softmax(layer_fc2)

        self.y_pred_cls = tf.argmax(self.y_pred, axis=1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=self.y_true)

        self.cost = tf.reduce_mean(cross_entropy)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.correct_prediction = tf.equal(self.y_pred_cls, y_true_cls)

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.session = tf.Session()

        self.saver = tf.train.Saver()

    def setup_single_variable(self, name="", default=0, set_to_default=False):

        if set_to_default:
                return default

        var = input(name + " ( default: \"" + str(default) + "\" ): ")
        if not var:
                var = default

        return var

    def setup_convolutional_layer(self, default_filter_size=0, default_num_filters=0, number=0, set_to_default=False):

        if set_to_default:
                return default_filter_size, default_num_filters

        print("----- CONVOLUTION LAYER %s -----" % number)

        filter_size = self.setup_single_variable(name="Filter size", default=default_filter_size)
        num_filters = self.setup_single_variable(name="# of filters", default=default_num_filters)

        return filter_size, num_filters

    def set_parameters(self, set_to_default):

        os.system('clear')

        print("----- SETUP PARAMETERS -----")

        # The number of pixels in each dimension of an image.
        self.img_size = self.setup_single_variable(name = "Image size", default=96, set_to_default=set_to_default)

        # Tuple with height and width of images used to reshape arrays.
        self.img_shape = (self.img_size, self.img_size)

        # The images are stored in one-dimensional arrays of this length.
        self.img_size_flat = self.img_shape[0] * self.img_shape[1]

        # Number of classes, one class for each of 5 actions.
        self.num_classes = self.setup_single_variable(name = "# of classes", default=5, set_to_default=set_to_default)

        # Learning rate.
        self.learning_rate = self.setup_single_variable(name = "Learning rate", default=0.00001, set_to_default=set_to_default)

        # Convolutional Layer 1.
        self.filter_size1, self.num_filters1 = self.setup_convolutional_layer(default_filter_size=3, default_num_filters=16, number=1, set_to_default=set_to_default)

        # Convolutional Layer 2.
        self.filter_size2, self.num_filters2 = self.setup_convolutional_layer(default_filter_size=3, default_num_filters=32, number=2, set_to_default=set_to_default)

        # Convolutional Layer 3.
        self.filter_size3, self.num_filters3 = self.setup_convolutional_layer(default_filter_size=3, default_num_filters=32, number=3, set_to_default=set_to_default)

        # Convolutional Layer 4.
        self.filter_size4, self.num_filters4 = self.setup_convolutional_layer(default_filter_size=3, default_num_filters=32, number=4, set_to_default=set_to_default)

        # Fully-connected layer.
        self.fc_size = self.setup_single_variable(name = "# of neurons in fully connected layer", default=128, set_to_default=set_to_default)

    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(self, length):
    	return tf.Variable(tf.constant(0.05, shape=[length]))

    def new_conv_layer(self,
                       input,              # The previous layer.
		               num_input_channels, # Num. channels in prev. layer.
		               filter_size,        # Width and height of each filter.
		               num_filters,        # Number of filters.
		               use_pooling=True):  # Use 2x2 max-pooling.

		# Shape of the filter-weights for the convolution.
		# This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]

		# Create new weights aka. filters with the given shape.
        weights = self.new_weights(shape=shape)

		# Create new biases, one for each filter.
        biases = self.new_biases(length=num_filters)

		# Create the TensorFlow operation for convolution.
		# Note the strides are set to 1 in all dimensions.
		# The first and last stride must always be 1,
		# because the first is for the image-number and
		# the last is for the input-channel.
		# But e.g. strides=[1, 2, 2, 1] would mean that the filter
		# is moved 2 pixels across the x- and y-axis of the image.
		# The padding is set to 'SAME' which means the input image
		# is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=input,
		                     filter=weights,
		                     strides=[1, 1, 1, 1],
		                     padding='SAME')

		# Add the biases to the results of the convolution.
		# A bias-value is added to each filter-channel.
        layer += biases

		# Use pooling to down-sample the image resolution?
        if use_pooling:
		    # This is 2x2 max-pooling, which means that we
		    # consider 2x2 windows and select the largest value
		    # in each window. Then we move 2 pixels to the next window.
                layer = tf.nn.max_pool(value=layer,
		                           ksize=[1, 2, 2, 1],
		                           strides=[1, 2, 2, 1],
		                           padding='SAME')

		# Rectified Linear Unit (ReLU).
		# It calculates max(x, 0) for each input pixel x.
		# This adds some non-linearity to the formula and allows us
		# to learn more complicated functions.
        layer = tf.nn.relu(layer)

		# Note that ReLU is normally executed before the pooling,
		# but since relu(max_pool(x)) == max_pool(relu(x)) we can
		# save 75% of the relu-operations by max-pooling first.

		# We return both the resulting layer and the filter-weights
		# because we will plot the weights later.
        return layer, weights

    def flatten_layer(self, layer):
		# Get the shape of the input layer.
        layer_shape = layer.get_shape()

		# The shape of the input layer is assumed to be:
		# layer_shape == [num_images, img_height, img_width, num_channels]

		# The number of features is: img_height * img_width * num_channels
		# We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()

		# Reshape the layer to [num_images, num_features].
		# Note that we just set the size of the second dimension
		# to num_features and the size of the first dimension to -1
		# which means the size in that dimension is calculated
		# so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])

		# The shape of the flattened layer is now:
		# [num_images, img_height * img_width * num_channels]

		# Return both the flattened layer and the number of features.
        return layer_flat, num_features

    def new_fc_layer(self,
                     input,          # The previous layer.
		             num_inputs,     # Num. inputs from prev. layer.
		             num_outputs,    # Num. outputs.
		             use_relu=True): # Use Rectified Linear Unit (ReLU)?

		# Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

		# Calculate the layer as the matrix multiplication of
		# the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

		# Use ReLU?
        if use_relu:
                layer = tf.nn.relu(layer)

        return layer

    def load(self, file_name):
        self.saver.restore(self.session, file_name)

    def save(self, file_name):
        self.saver.save(self.session, file_name)
