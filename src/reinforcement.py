import numpy as np
from keras import initializers
from keras.initializers import normal, identity
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import RMSprop, SGD, Adam
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten
from src.features import *
from src.metrics import *
from src.utils import *

# Different actions that the agent can do
number_of_actions = 6
# Actions captures in the history vector
actions_of_history = 4
# Visual descriptor size
visual_descriptor_size = 25088
# Reward movement action
reward_movement_action = 1
# Reward terminal action
reward_terminal_action = 3
# IoU required to consider a positive detection
iou_threshold = 0.5


def update_history_vector(history_vector, action):
    action_vector = np.zeros(number_of_actions)
    action_vector[action-1] = 1
    size_history_vector = np.size(np.nonzero(history_vector))
    updated_history_vector = np.zeros(number_of_actions*actions_of_history)
    if size_history_vector < actions_of_history:
        aux2 = 0
        for l in range(number_of_actions*size_history_vector, number_of_actions*size_history_vector+number_of_actions - 1):
            history_vector[l] = action_vector[aux2]
            aux2 += 1
        return history_vector
    else:
        for j in range(0, number_of_actions*(actions_of_history-1) - 1):
            updated_history_vector[j] = history_vector[j+number_of_actions]
        aux = 0
        for k in range(number_of_actions*(actions_of_history-1), number_of_actions*actions_of_history):
            updated_history_vector[k] = action_vector[aux]
            aux += 1
        return updated_history_vector


def get_image_vector(image, model):
    return np.array(get_conv_image_descriptor_for_image(image, model))

def get_state(target_image, search_image, history_vector):
    history_vector = np.reshape(history_vector, (number_of_actions*actions_of_history, 1))
    target_image = np.reshape(target_image, (-1, 1))
    search_image = np.reshape(search_image, (-1, 1))
    state = np.vstack((target_image, search_image, history_vector))
    return state
    


def get_state_pool45(history_vector,  region_descriptor):
    history_vector = np.reshape(history_vector, (24, 1))
    return np.vstack((region_descriptor, history_vector))


def get_reward_movement(iou, new_iou):
    return new_iou-iou


def get_reward_trigger(new_iou):
    if new_iou > iou_threshold:
        reward = new_iou
    else:
        reward = -1
    return reward


def get_q_network(weights_path):
    model = Sequential()
    model.add(Dense(1024, kernel_initializer=lambda shape:K.random_normal(shape), input_shape=(8216,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, kernel_initializer=lambda shape:K.random_normal(shape)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_actions, kernel_initializer=lambda shape:K.random_normal(shape)))
    model.add(Activation('linear'))
    adam = Adam(lr=1e-6)
    model.compile(loss='mse', optimizer=adam)
    if weights_path != "0":
        model.load_weights(weights_path)
    model.summary()
    return model


def get_array_of_q_networks_for_pascal(weights_path, class_object):
    q_networks = []
    if weights_path == "0":
        for i in range(20):
            q_networks.append(get_q_network("0"))
    else:
        for i in range(20):
            if i == (class_object-1):
                q_networks.append(get_q_network(weights_path + "/model" + str(i) + "h5"))
            else:
                q_networks.append(get_q_network("0"))
    return np.array([q_networks])

def do_action(action,search_image, region_image, region_mask, offset, size_mask, scale_mask):
    if action == 1:
        offset_aux = (0, 0)
    elif action == 2:
        offset_aux = (0, size_mask[1] * scale_mask)
        offset = (offset[0], offset[1] + size_mask[1] * scale_mask)
    elif action == 3:
        offset_aux = (size_mask[0] * scale_mask, 0)
        offset = (offset[0] + size_mask[0] * scale_mask, offset[1])
    elif action == 4:
        offset_aux = (size_mask[0] * scale_mask, 
                      size_mask[1] * scale_mask)
        offset = (offset[0] + size_mask[0] * scale_mask,
                  offset[1] + size_mask[1] * scale_mask)
    elif action == 5:
        offset_aux = (size_mask[0] * scale_mask / 2,
                      size_mask[0] * scale_mask / 2)
        offset = (offset[0] + size_mask[0] * scale_mask / 2,
                  offset[1] + size_mask[0] * scale_mask / 2)
    region_image = region_image[int(offset_aux[0]):int(offset_aux[0]) + int(size_mask[0]),int(offset_aux[1]):int(offset_aux[1]) + int(size_mask[1])]
    region_mask[int(offset[0]):int(offset[0]) + int(size_mask[0]), int(offset[1]):int(offset[1]) + int(size_mask[1])] = 1
    region_image_from_original = search_image[int(offset[0]):int(offset[0]) + int(size_mask[0]), int(offset[1]):int(offset[1]) + int(size_mask[1])]
    return region_image_from_original, region_image, region_mask