import tensorflow as tf

# Convolutional Layer 1.
filter_size1 = 3          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 3          # Convolution filters are 5 x 5 pixels.
num_filters2 = 32         # There are 36 of these filters.

# Convolutional Layer 3.
filter_size3 = 3          # Convolution filters are 5 x 5 pixels.
num_filters3 = 32         # There are 36 of these filters.

# Convolutional Layer 4.
filter_size4 = 3          # Convolution filters are 5 x 5 pixels.
num_filters4 = 32         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# The number of pixels in each dimension of an image.
img_size = 96

# The images are stored in one-dimensional arrays of this length.
#img_size_flat = data.img_size_flat

# Tuple with height and width of images used to reshape arrays.
img_shape = (96, 96)

# Number of classes, one class for each of 5 actions.
num_classes = 5

# Number of colour channels for the images: 1 channel for gray-scale.
#num_channels = data.num_channels

class Model:
    def __init__(self, history_length = 1, learning_rate = 3e-4, batch_size = 1):

        self.learning_rate = learning_rate

		self.x_input = tf.placeholder(dtype=tf.float32,
									 shape = [None, img_shape[0], img_shape[1], history_length],
									 name = "x_input")

		self.x_image = tf.reshape(x, [-1, img_size, img_size, history_length])

        self.y_label = tf.placeholder(dtype=tf.float32,
									 shape = [None, num_classes],	
									 name = "y_label")

		self.y_true = tf.placeholder(tf.float32,
									 shape=[None, num_classes],
									 name='y_true')

		self.y_true_cls = tf.argmax(y_true, axis=1)

		self.layer_conv1, self.weights_conv1 = \
			new_conv_layer(input=x_image,
				           num_input_channels=history_length,
				           filter_size=filter_size1,
				           num_filters=num_filters1,
				           use_pooling=True)

		self.layer_conv2, self.weights_conv2 = \
			new_conv_layer(input=self.layer_conv1,
				           num_input_channels=num_filters1,
				           filter_size=filter_size2,
				           num_filters=num_filters2,
				           use_pooling=True)

		self.layer_conv3, self.weights_conv3 = \
					new_conv_layer(input=layer_conv2,
								   num_input_channels=num_filters2,
								   filter_size=filter_size3,
								   num_filters=num_filters3,
								   use_pooling=True)

		self.layer_conv4, self.weights_conv4 = \
					new_conv_layer(input=layer_conv3,
								   num_input_channels=num_filters3,
								   filter_size=filter_size4,
								   num_filters=num_filters4,
								   use_pooling=True)

		self.layer_flat, self.num_features = flatten_layer(layer_conv4)


		self.layer_fc1 = new_fc_layer(input=layer_flat,
                         			 num_inputs=num_features,
                                     num_outputs=fc_size,
                                     use_relu=True)

		self.layer_fc2 = new_fc_layer(input=layer_fc1,
				                     num_inputs=fc_size,
				                     num_outputs=num_classes,
				                     use_relu=False)

		self.y_pred = tf.nn.softmax(layer_fc2)
		self.y_pred_cls = tf.argmax(y_pred, axis=1)

		self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        			labels=y_true)

		self.cost = tf.reduce_mean(cross_entropy)

		self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

		self.correct_prediction = tf.equal(y_pred_cls, y_true_cls)

		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def new_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

	def new_biases(length):
    	return tf.Variable(tf.constant(0.05, shape=[length]))

	def new_conv_layer(input,              # The previous layer.
		               num_input_channels, # Num. channels in prev. layer.
		               filter_size,        # Width and height of each filter.
		               num_filters,        # Number of filters.
		               use_pooling=True):  # Use 2x2 max-pooling.

		# Shape of the filter-weights for the convolution.
		# This format is determined by the TensorFlow API.
		shape = [filter_size, filter_size, num_input_channels, num_filters]

		# Create new weights aka. filters with the given shape.
		weights = new_weights(shape=shape)

		# Create new biases, one for each filter.
		biases = new_biases(length=num_filters)

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

	def flatten_layer(layer):
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

	def new_fc_layer(input,          # The previous layer.
		             num_inputs,     # Num. inputs from prev. layer.
		             num_outputs,    # Num. outputs.
		             use_relu=True): # Use Rectified Linear Unit (ReLU)?

		# Create new weights and biases.
		weights = new_weights(shape=[num_inputs, num_outputs])
		biases = new_biases(length=num_outputs)

		# Calculate the layer as the matrix multiplication of
		# the input and weights, and then add the bias-values.
		layer = tf.matmul(input, weights) + biases

		# Use ReLU?
		if use_relu:
		    layer = tf.nn.relu(layer)

		return layer

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)
