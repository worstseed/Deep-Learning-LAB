from __future__ import print_function

import numpy as np
import tensorflow as tf
import datetime

from model import Model
from utils import *
from utils_data import *
from tensorboard_evaluation import Evaluation
from random import randint

def train_model(X_train, y_train,
                X_valid, y_valid,
                epochs, batch_size,
                lr,
                history_length,
                set_to_default,
                model_dir = "./models", tensorboard_dir = "./tensorboard"):

    file_path = "/models/" + str(history_length) + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    current_directory = os.getcwd()
    parent_directory = os.path.split(current_directory)[0]
    # model_dir = os.path.join(parent_directory, file_path)
    model_dir = str(parent_directory) + file_path

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print("... folders created")

    when_to_show = int(input("Show every x iteration: "))

    X_train = reshape_input_data(X_train, history_length)
    X_valid = reshape_input_data(X_valid, history_length)
    print("... input data reshaped")

    agent = Model(batch_size = batch_size, history_length = history_length, set_to_default = set_to_default)
    print("... model created")

    # Initialization
    agent.session.run(tf.global_variables_initializer())
    tensorboard_eval = Evaluation(tensorboard_dir)
    tf.reset_default_graph()
    print("... model initialized")

    training_accuracy = np.zeros((epochs))
    validation_accuracy = np.zeros((epochs))

    os.system('clear')
    print("... train model")
    # training loop
    for i in range(epochs):

        first_index, last_index = get_minibatch_indices(X_train.shape[0], batch_size, history_length)

        X_train_mini = X_train[first_index : last_index, :, :, :]
        y_train_mini = y_train[first_index : last_index, :]

        feed_dict_train = {agent.x: X_train_mini, agent.y_true: y_train_mini}
        agent.session.run(agent.optimizer, feed_dict=feed_dict_train)

        if i % when_to_show == 0:

            # Calculate the accuracy on the training-set.
            training_accuracy[i] += agent.session.run(agent.accuracy, feed_dict=feed_dict_train)
            feed_dict_valid = {agent.x: X_valid, agent.y_true: y_valid}
            validation_accuracy[i] += agent.session.run(agent.accuracy, feed_dict=feed_dict_valid)

            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%} \n"
            print(msg.format(i + 1, training_accuracy[i], validation_accuracy[i]))

            log_dir = "./log/" + str(history_length) + "/"
            save_path = os.path.join(model_dir, "log")
            full_path = save_path + '.txt'
            f = open(full_path, "a")
            f.write(msg.format(i + 1, training_accuracy[i], validation_accuracy[i]))

            eval_dict = {"train":training_accuracy[i], "valid":validation_accuracy[i]}
            tensorboard_eval.write_episode_data(i, eval_dict)

    # save your agent
    save_path = os.path.join(model_dir, "agent.ckpt")
    agent.save(save_path)
    print("... model saved in file: %s" % save_path)
    agent.session.close()

if __name__ == "__main__":

    history_length = int(input("Set history length (>= 1): "))

    X_train, y_train, X_valid, y_valid = read_data("./data")
    print("... data read")

    data_used = X_train.shape[0] // 3

    if history_length == 1:
      X_train, y_train = shuffle_data(X_train, y_train)
      X_valid, y_valid = shuffle_data(X_valid, y_valid)
      print("... data shuffled")

    X_train, y_train_hot, X_valid, y_valid_hot = preprocessing(X_train, y_train, X_valid, y_valid, history_length = history_length)
    print("... data preprocessed")

    X_train, y_train_hot = uniform_sampling(X_train, y_train_hot, data_used, history_length)
    print("... performed uniform sampling")

    train_model(X_train, y_train_hot,
                X_valid, y_valid_hot,
                history_length = history_length,
                set_to_default = True,
                epochs = 1000, batch_size = 256, lr = 0.0001)
