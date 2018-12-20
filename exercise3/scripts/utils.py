import numpy as np
import random
import os

LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE = 3
BRAKE = 4

def one_hot(labels):
    """
    this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    # n_classes = classes.size
    n_classes = 5 # CHEAT

    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels

def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32')

def action_to_id(a):
    """
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]): return LEFT               # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]): return RIGHT             # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]): return ACCELERATE        # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.2]): return BRAKE             # BRAKE: 4
    # elif all(a == [0.0, 0.0, 0.0]): return ACCELERATE if random.random() < 0.03 else STRAIGHT
    else:
        return STRAIGHT                                      # STRAIGHT = 0

def make_it_hot(y, dtype = np.int8):

    length = y.shape[0]
    temp = np.zeros(length, dtype)

    for i in range(length):
        temp[i] = action_to_id(y[i])

    return one_hot(temp)

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

def list_subdirs_in_dir(dirname = "."):

    for x in os.walk(dirname):
        print(x[0])
