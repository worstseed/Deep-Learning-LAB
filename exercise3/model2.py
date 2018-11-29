import tensorflow as tf

class Model:

    def __init__(
                self,
                image_width = 96,
                image_height = 96,
                n_classes = 5,
                history_length = 1,
                conv_layers = None,
                lstm_layers = None,
                name = '',
                learning_rate = 0.1,
                path = './models/'):

        self.name = name
        self.savepath = path

        if conv_layers is None:
            conv_layers = [
                {
                    'name'        : 'conv1',
                    'filters'     : 16     ,
                    'kernel size' : 5      ,
                    'padding'     : 'SAME' ,
                    'stride'      : (1, 1) ,
                    'activation'  : tf.nn.relu,

                    'pooling'     : 'max',
                    'pool ksize'  : [1, 2, 2, 1],
                    'pool stride' : [1, 2, 2, 1],
                    'pool padding': 'VALID'
                },

                {
                    'name'       : 'conv2',
                    'filters'    : 24,
                    'kernel size': 3,
                    'padding'    : 'SAME',
                    'stride'     : (1, 1),
                    'activation' : tf.nn.relu,

                    'pooling'     : 'max',
                    'pool ksize'  : [1, 2, 2, 1],
                    'pool stride' : [1, 2, 2, 1],
                    'pool padding': 'VALID'
                }
            ]


        if lstm_layers is None:
            lstm_layers = [
                {
                    'name'        : 'lstm1',
                    'units'       : 16     ,
                    'kernel size' : 5      ,
                    'padding'     : 'SAME' ,
                    'stride'      : (1,1)  ,
                    'activation'  : tf.nn.relu
                }
            ]


        # MODEL
        # ================================================================================

        # Input Layer
        with tf.name_scope("inputs"):
            self.X = tf.placeholder(tf.float32,
                               shape = [None, image_width, image_height, history_length],
                               name  = "x"
                              )
            self.y = tf.placeholder(tf.float32,
                               shape = [None, n_classes],
                               name  = "y"
                              )


        # Convolutional Layers
        conv_input = self.X
        for layer in conv_layers:
            with tf.name_scope(layer['name']):
                conv_input = tf.layers.conv2d(conv_input,
                                              filters     = layer['filters'],
                                              kernel_size = layer['kernel size'],
                                              strides     = layer['stride'],
                                              padding     = layer['padding'],
                                              activation  = layer['activation'],
                                              name        = layer['name']
                                             )
            if 'pooling' in layer:
                with tf.name_scope(layer['name'] + '_pool'):
                    if layer['pooling'] == 'max':
                        conv_input = tf.nn.max_pool(conv_input,
                                                    ksize   = layer['pool ksize'],
                                                    strides = layer['pool stride'],
                                                    padding = layer['pool padding'],
                                                   )

        if conv_layers != []:
            # Flatten the layer
            last_shape = conv_input.get_shape().as_list()
            flat_size = last_shape[1] * last_shape[2] * last_shape[3]
            conv_input = tf.reshape(conv_input, shape=[-1,  flat_size])

        # Recurrent Layers
        lstm_input = conv_input
        for layer in lstm_layers:
            with tf.name_scope(layer['name']):

                if tf.test.is_gpu_available(cuda_only = True):
                    print('GPU Available, using CUDA optimized LSTM')
                    lstm_cell = tf.contrib.cudnn_rnn.CudnnLSTM(layer['num layers'],
                                                               layer['num units']
                                                              )

                else:
                    lstm_cell = tf.contrib.rnn.LSTMBlockFusedCell(layer['num units'])

                    lstm_input, new_states = tf.nn.dynamic_rnn(lstm_cell, lstm_input)


        # Dense Layers
        dense_input = lstm_input

        dense_input = tf.layers.dense(
                                    dense_input,
                                    100,
                                    activation=tf.nn.relu,
                                    name="fc1")

        dense_input = tf.layers.dense(
                                    dense_input,
                                    30,
                                    activation = tf.nn.relu,
                                    name = "fc2")


        # Output Layer
        with tf.name_scope("output"):
            self.logits = tf.layers.dense(
                                        dense_input,
                                        n_classes,
                                        name = "output")

            self.Y_proba = tf.nn.softmax(self.logits, name = "Y_proba")

        # LOSS AND OPTIMIZER
        # ================================================================================
        with tf.name_scope("train"):
            xentropy = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.y)
            self.loss = tf.reduce_mean(xentropy)
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            self.trainer = self.optimizer.minimize(self.loss)


        with tf.name_scope("eval"):
            correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.logits, 1))
            self.accuracyuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # TENSORFLOW SESSION
        # ================================================================================
        self.init = tf.global_variables_initializer()

        self.session = tf.Session()

        self.saver = tf.train.Saver()

    def evaluate(self, x, y, max_batch_size = 1000):

        accuracy = 0
        loss = 0

        batch_size = x.shape[0]
        num_iter = batch_size // max_batch_size

        if batch_size % max_batch_size != 0:
            num_iter += 1

        for i in range(num_iter):

            if (i + 1) * max_batch_size > batch_size:
                end = batch_size
                count = batch_size - (i * max_batch_size)
            else:
                end = (i + 1) * max_batch_size
                count = max_batch_size

            accuracy  += self.accuracy.eval(feed_dict = {
                self.X: x[i * max_batch_size: end],
                self.y: y[i * max_batch_size: end]
            },
                session = self.session) * count
            loss += self.loss.eval(feed_dict = {
                self.X: x[i * max_batch_size: end],
                self.y: y[i * max_batch_size: end]
            },
                session = self.session) * count


        return loss / batch_size, accuracy / batch_size

    def predict(self, x):
        return self.Y_proba.eval(feed_dict = {self.X: x}, session = self.session)

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)
