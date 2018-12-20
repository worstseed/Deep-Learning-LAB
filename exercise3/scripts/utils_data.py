import os
import gzip
import pickle
import numpy as np
from utils import *

def read_data(datasets_dir = "./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    current_directory = os.getcwd()
    parent_directory = os.path.split(current_directory)[0]
    data_file = os.path.join(parent_directory, datasets_dir, 'data.pkl.gzip')

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

def preprocessing(X_train, y_train, X_valid, y_valid, history_length, onehot = True):

    print("... preprocessing data")

    X_train, X_valid = input_data_to_grayscale(X_train, X_valid)

    if onehot:
        y_train, y_valid = output_data_to_discrete(y_train, y_valid)

    if history_length > 1:
        X_train, y_train, X_valid, y_valid = use_history(history_length, X_train, y_train, X_valid, y_valid)

    return X_train, y_train, X_valid, y_valid

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

  X = np.zeros((X_train.shape[0] - history_length, X_train.shape[1], X_train.shape[2], history_length))
  y = y_train[history_length:]
  X_v = np.zeros((X_valid.shape[0] - history_length, X_valid.shape[1], X_valid.shape[2], history_length))
  y_v = y_valid[history_length:]

  for i in range(history_length):
      X[:, :, :, i] = np.reshape(X_train[i:X_train.shape[0] - history_length + i], (X_train.shape[0] - history_length, 96, 96))
      X_v[:, :, :, i] = np.reshape(X_valid[i:X_valid.shape[0] - history_length + i], (X_valid.shape[0] - history_length, 96, 96))

  return X, y, X_v, y_v

def uniform_sampling(X, y, num_samples, history_length):

  n = y.shape[0] // history_length
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

  weights[straight] = straight_weight
  weights[left] = left_weight
  weights[right] = right_weight
  weights[accelerate] = accelerate_weight
  weights[brake] = brake_weight

  weights /= np.sum(weights)

  zipped = np.array(list(zip(X ,y)))

  chosen_indices = np.random.choice(np.arange(n), num_samples // history_length, replace = False, p = weights)

  zipped = zipped[chosen_indices]

  X, y = zip(*zipped)

  return np.array(X), np.array(y)

def reshape_to_history_length(x, history_length):

    batch_size   = x.shape[0]
    image_width = x.shape[1]
    image_height   = x.shape[2]

    temp = np.empty((batch_size - history_length + 1, image_width, image_height, history_length))

    for i in range(batch_size - history_length):
        temp[i, :, :, :] = np.transpose(x[i: i + history_length, :, :], (1, 2, 0))

    return temp

def get_minibatch_indices(data_size, batch_size, history_length):

    first_index = np.random.randint(0, data_size - batch_size - history_length - 1)
    return first_index, (first_index + batch_size)

def reshape_input_data(x, history_length):

    return np.reshape(x, (x.shape[0], 96, 96, history_length))
