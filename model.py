from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout
from keras.applications import InceptionV3

import pandas as pd
import numpy as np
import cv2

def load_data_from_log(driving_log):
    data = pd.read_csv(driving_log, dtype={'center': str, 'left': str, 'right': str,
                                           'steering': np.float32, 'throttle': np.float32,
                                           'brake': np.float32, 'speed': np.float32})

    speed = np.array(data.speed)
    steering = np.array(data.steering)
    throttle = np.array(data.throttle)
    brake = np.array(data.brake)
    center = np.array(data.center)
    left = np.array(data.left)
    right = np.array(data.right)

    return steering, throttle, brake, speed, center, left, right

def Net():
    model = InceptionV3(include_top=False, weights='imagenet', input_shape=(160, 320, 3))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout)
    model.add(Dense(2048, ))


    model.add(Activation('sigmoid'))

    return model


folder_data = '../data/'
steering, throttle, brake, speed, center, left, right = load_data_from_log(folder_data + 'driving_log.csv')

test_image = cv2.imread(folder_data+center[0])

test_image.shape
