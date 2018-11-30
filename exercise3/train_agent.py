from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import tensorflow as tf

from model import Model
from utils import *
from tensorboard_evaluation import Evaluation

def read_data(datasets_dir = "./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')

    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1 - frac) * n_samples)], y[:int((1 - frac) * n_samples)]
    X_valid, y_valid = X[int((1 - frac) * n_samples):], y[int((1 - frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid

# useful function, we don't really need that, but it helps.
# borrowed from JAB.
def plot_data(x, y, history_length = 1, rows = 10, title = ''):

    image_path = './img/'

    fig = plt.figure(1)
    fig.suptitle(title)
    plt.subplots_adjust(hspace = 1)

    rows = rows
    columns = history_length

    spi = 1

    for i in range(rows):
        for j in range(columns):
            subplot = fig.add_subplot(rows, columns, spi)
            subplot.imshow(x[i, :, :, j], cmap='gray')

            subplot.set_xticks([])
            subplot.set_yticks([])

            if j == columns - 1:
                subplot.set(xlabel='A: ' + ACTIONS[action_to_id(y[i])]['log'])

            spi += 1

    fig.show()

    fig.savefig(image_path + title + '.png', dpi=300)

def reshape_to_history_length(x, history_length):

    batch_size   = x.shape[0]
    image_width = x.shape[1]
    image_height   = x.shape[2]

    temp = np.empty((batch_size - history_length + 1, image_width, image_height, history_length))

    for i in range(batch_size - history_length):
        temp[i, :, :, :] = np.transpose(x[i: i + history_length, :, :, 0], (1, 2, 0))

    return tmep

def make_it_hot(y, dtype = np.int8):

    length = y.shape[0]
    temp = np.zeros(length, dtype)

    for i in range(length):
        temp[i] = action_to_id(y[i])

    return one_hot(temp)

def input_data_to_grayscale(X_train, X_valid):
    # convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    return rgb2gray(X_train), rgb2gray(X_valid)

def output_data_to_discrete(y_train, y_valid):
    # you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot()
    #    useful and you may want to return X_train_unhot ... as well.
    return make_it_hot(y_train), make_it_hot(y_valid)

def use_history(history_length, X_train, y_train, X_valid, y_valid):
    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    X_train = reshape_to_history_length(X_train, history_length)
    y_train[history_length] = y_train[history_length - 1:]
    X_valid = reshape_to_history_length(X_valid, history_length)
    y_valid[history_length] = y_valid[history_length - 1:]

    return X_train, y_train, X_valid, y_valid

def preprocessing(X_train, y_train, X_valid, y_valid, history_length = 1, onehot = True):

    print("... preprocessing data")

    X_train, X_valid = input_data_to_grayscale(X_train, X_valid)

    if onehot:
        y_train, y_valid = output_data_to_discrete(y_train, y_valid)

    if history_length > 1:
        X_train, y_train, X_valid, y_valid = use_history(history_length, X_train, y_train, X_valid, y_valid)

    return X_train, y_train, X_valid, y_valid

def get_minibatch_indices(data_size, batch_size, history_length):

    first_index = np.random.randint(0, data_size - batch_size - history_length - 1)
    return first_index, (first_index + batch_size)

def reshape_input_data(x):
    return np.reshape(x, (x.shape[0], 96, 96, 1))

# def train_model(X_train, y_train,
#                 X_valid, y_valid,
#                 n_minibatches,
#                 batch_size,
#                 lr,
#                 model_dir = "./models",
#                 tensorboard_dir = "./tensorboard"):
def train_model(X_train, y_train,
                X_valid, y_valid,
                n_minibatches, batch_size,
                lr,
                history_length=1,
                model_dir = "./models", tensorboard_dir = "./tensorboard"):

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print("... folders created")

    X_train = reshape_input_data(X_train)
    X_valid = reshape_input_data(X_valid)
    print("... input data reshaped")

    # when_to_show = int(input("Show every x iteration: "))

    agent = Model(batch_size = batch_size, learning_rate = lr, history_length = history_length)
    print("... model created")

    # tensorboard_eval = Evaluation(tensorboard_dir)

    # Initialization
    agent.sess.run(tf.global_variables_initializer())
    tf.reset_default_graph()
    print("... model initialized")

    training_cost = np.zeros((n_minibatches))
    validation_cost = np.zeros((n_minibatches))

    print("... train model")
    # training loop
    for i in range(n_minibatches):

        first_index, last_index = get_minibatch_indices(X_train.shape[0], batch_size, history_length)

        X_train_mini = X_train[first_index : last_index, :, :, :]
        y_train_mini = y_train[first_index : last_index, :]

        training_cost[i] += agent.sess.run(agent.cost, feed_dict={agent.x_input: X_train_mini, agent.y_label: y_train_mini})
        validation_cost[i] += agent.sess.run(agent.cost, feed_dict={agent.x_input: X_train_mini, agent.y_label: y_train_mini})
        # training_accuracy[i] += agent.sess.run(agent.accuracy, feed_dict={agent.x_input: X_train_mini, agent.y_label: y_train_mini})
        # validation_accuracy[i] += agent.sess.run(agent.accuracy, feed_dict={agent.x_input: X_train_mini, agent.y_label: y_train_mini})
        # agent.session.run(agent.trainer, feed_dict = {agent.X: X_train_mini, agent.y: y_train_mini})
        # train_cost += agent.sess.run(agent.cost, feed_dict={agent.x_input: X_train_mini, agent.y_label: y_train_mini})

        # compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
        #    your training in your web browser
        # if (i % when_to_show == 0):
            # train_loss, train_accuracy = agent.evaluate(X_train, y_train)
            # valid_loss, valid_accuracy = agent.evaluate(X_valid, y_valid)

            # print(  "Minibatch: ", i ,
            #         " Train accuracy: ", train_accuracy,
            #         " Train Loss: ", train_loss,
            #         ", Test accuracy: ", valid_accuracy,
            #         " Test Loss: ", valid_loss)

            # agent.save('i' + str(i) + '_TrainAccuracy_' + "{:.4f}".format(train_accuracy * 100))
        print("[%d/%d]: training_cost: %.2f, validation_cost: %.2f" %(i+1, n_minibatches, 100*training_cost[i], 100*validation_cost[i]))
        # print("[%d/%d]: training_accuracy: %.2f, validation_accuracy: %.2f" %(i+1, epochs, 100*training_accuracy[i], 100*validation_accuracy[i]))
        # eval_dict = {"train":training_cost[i], "valid":validation_cost[i]}
        # tensorboard_eval.write_episode_data(i, eval_dict)
        # tensorboard_eval.write_episode_data(...  # TODO: implement the training)


    # save your agent
    save_path = os.path.join(model_dir, "agent.ckpt")
    agent.save(save_path)
    print("... model saved in file: %s" % save_path)
    agent.sess.close()

if __name__ == "__main__":

    history_length = int(input("Set history length (>= 1): "))

    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")
    print("... data read")

    # preprocess data
    X_train, y_train_hot, X_valid, y_valid_hot = preprocessing(X_train, y_train, X_valid, y_valid, history_length = history_length)
    print("... data preprocessed")

    # # Plot preprocessed data for debugging - still JAB
    # plot_data(X_train, y_train, history_length, history_length + 5, 'Sample Train Data')
    # plot_data(X_train, y_train, history_length, history_length + 5, 'Sample Validation Data')

    train_model(X_train, y_train,
                X_valid, y_valid,
                history_length = history_length,
                n_minibatches = 10000, batch_size = 64, lr = 0.0004)
