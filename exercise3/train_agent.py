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

    print("Size of:")
    print("- Training-set:\t\t{}".format(X_train.shape[0]))
    print("- Validation-set:\t{}".format(X_valid.shape[0]))

    return X_train, y_train, X_valid, y_valid

def shuffle_data(X, y):

    zipped = np.array(list(zip(X ,y)))
    np.random.shuffle(zipped)
    X, y = zip(*zipped)
    X = np.array(X)
    y = np.array(y)

    return X, y

def reshape_to_history_length(x, history_length):

    batch_size   = x.shape[0]
    image_width = x.shape[1]
    image_height   = x.shape[2]

    temp = np.empty((batch_size - history_length + 1, image_width, image_height, history_length))

    for i in range(batch_size - history_length):
        temp[i, :, :, :] = np.transpose(x[i: i + history_length, :, :, 0], (1, 2, 0))

    return temp

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
def uniform_sampling(X, y, num_samples, history_length = 1, use_history = False):
  
  n = y.shape[0]
  weights = np.zeros(n)
  
  straight = []
  left = []
  right = []
  accelerate = []
  brake = []
  
  for i in range(n):
    if (y[i] == [1., 0., 0., 0., 0.]).all():
      straight.append(i)
    elif (y[i] == [0., 1., 0., 0., 0.]).all():
      left.append(i)
    elif (y[i] == [0., 0., 1., 0., 0.]).all():
      right.append(i)
    elif (y[i] == [0., 0., 0., 1., 0.]).all():
      accelerate.append(i)
    elif (y[i] == [0., 0., 0., 0., 1.]).all():
      brake.append(i)
  
  straight_weight = n / len(straight) if (len(straight) != 0) else 1
  left_weight = n / len(left) if (len(left) != 0) else 1
  right_weight = n / len(right) if (len(right) != 0) else 1
  accelerate_weight = n / len(accelerate) if (len(accelerate) != 0) else 1
  brake_weight = n / len(brake) if (len(brake) != 0) else 1
  
  sum_weight = straight_weight + left_weight + right_weight + accelerate_weight + brake_weight
  straight_weight /= sum_weight
  left_weight /= sum_weight
  right_weight /= sum_weight
  accelerate_weight /= sum_weight
  brake_weight /= sum_weight
  
#   print("straight weight: ", straight_weight)
#   print("left weight: ", left_weight)
#   print("right weight: ", right_weight)
#   print("accelerate weight: ", accelerate_weight)
#   print("brake weight: ", brake_weight)

  weights[straight] = straight_weight
  weights[left] = left_weight
  weights[right] = right_weight
  weights[accelerate] = accelerate_weight
  weights[brake] = brake_weight

  weights /= np.sum(weights)
    
  zipped = np.array(list(zip(X ,y)))
    
  chosen_indices = np.random.choice(np.arange(n), num_samples, replace = False, p = weights)
  
  if use_history:
    chosen_indices = np.array([range(i - history_length, i, 1) for i in chosen_indices]).flatten() 
  
  zipped = zipped[chosen_indices]
      
  X, y = zip(*zipped)

  return np.array(X), np.array(y)

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

def count_output_data_hot_instances(y, j = ''):

    counter_s = 0
    counter_l = 0
    counter_r = 0
    counter_a = 0
    counter_b = 0

    for i in range(y.shape[0]):
        if (y[i] == [1., 0., 0., 0., 0.]).all():
            counter_s += 1
        if (y[i] == [0., 1., 0., 0., 0.]).all():
            counter_l += 1
        if (y[i] == [0., 0., 1., 0., 0.]).all():
            counter_r += 1
        if (y[i] == [0., 0., 0., 1., 0.]).all():
            counter_a += 1
        if (y[i] == [0., 0., 0., 0., 1.]).all():
            counter_b += 1

    print("----- OUTPUT DATA -----")
    if (j != ''):
        print("------- ", j, "--------")
    print("STRAIGHT: ", counter_s)
    print("LEFT: ", counter_l)
    print("RIGHT: ", counter_r)
    print("ACCELERATE: ", counter_a)
    print("BRAKE: ", counter_b)
    print("-----------------------")

def train_model(X_train, y_train,
                X_valid, y_valid,
                epochs, batch_size,
                lr,
                history_length = 1,
                model_dir = "./models", tensorboard_dir = "./tensorboard"):

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print("... folders created")

    when_to_show = int(input("Show every x iteration: "))

    X_train = reshape_input_data(X_train)
    X_valid = reshape_input_data(X_valid)
    print("... input data reshaped")

    #y_train = np.zeros((y_train_hot.shape[0], 3))

    #for i in range(y_train_hot.shape[0]):
    #    if (y_train_hot[i] == [1., 0., 0., 0., 0.]).all():
    #        y_train[i] = [0.0, 0.0, 0.0] 
    #    if (y_train_hot[i] == [0., 1., 0., 0., 0.]).all():
    #        y_train[i] = [-1.0, 0.0, 0.0] 
    #    if (y_train_hot[i] == [0., 0., 1., 0., 0.]).all():
    #        y_train[i] = [1.0, 0.0, 0.0] 
    #    if (y_train_hot[i] == [0., 0., 0., 1., 0.]).all():
    #        y_train[i] = [0.0, 1.0, 0.0] 
    #    if (y_train_hot[i] == [0., 0., 0., 0., 1.]).all():
    #        y_train[i] = [0.0, 0.0, 0.2] 


    agent = Model(batch_size = batch_size, history_length = history_length)
    print("... model created")

    # tensorboard_eval = Evaluation(tensorboard_dir)

    # Initialization
    agent.session.run(tf.global_variables_initializer())
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

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            msg_valid = "Optimization Iteration: {0:>6}, Validation Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, training_accuracy[i]))
            print(msg_valid.format(i + 1, validation_accuracy[i]))
            #if i % (5 * when_to_show) == 0:
             #   count_output_data_hot_instances(y_train_mini, i)

        # compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
        #    your training in your web browser
            
        # eval_dict = {"train":training_cost[i], "valid":validation_cost[i]}
        # tensorboard_eval.write_episode_data(i, eval_dict)

    # save your agent
    save_path = os.path.join(model_dir, "agent.ckpt")
    agent.save(save_path)
    print("... model saved in file: %s" % save_path)
    agent.session.close()

if __name__ == "__main__":

    history_length = int(input("Set history length (>= 1): "))
    use_h = False

    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")
    print("... data read")

    offset = int(X_train.shape[0] / 3)

    if not use_h:
      X_train, y_train = shuffle_data(X_train, y_train)
      X_valid, y_valid = shuffle_data(X_valid, y_valid)
      print("... data shuffled")

    # preprocess data
    X_train, y_train_hot, X_valid, y_valid_hot = preprocessing(X_train, y_train, X_valid, y_valid, history_length = history_length)
    print("... data preprocessed")

    X_train, y_train_hot = uniform_sampling(X_train, y_train_hot, offset, history_length, use_history = use_h)
       
    #count_output_data_hot_instances(y_train_hot)

    train_model(X_train, y_train_hot,
                X_valid, y_valid_hot,
                history_length = history_length,
                epochs = 10000, batch_size = 64, lr = 0.0004)
