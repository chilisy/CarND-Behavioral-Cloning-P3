from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.applications import InceptionV3
from keras.models import Sequential

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
    model = Sequential()
    model.add(InceptionV3(include_top=False, weights='imagenet', input_shape=(160, 320, 3)))
    model.layers[0].trainable = False

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(512))
    model.add(Dense(48))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


folder_data = 'data/'
steering, throttle, brake, speed, center, left, right = load_data_from_log(folder_data + 'driving_log.csv')

test_img = cv2.imread(folder_data+center[0])
img_data = np.zeros_like(test_img)
img_data = np.expand_dims(img_data, axis=0)
for img_name in center:
    img = cv2.imread(folder_data+img_name)
    img = np.expand_dims(img, axis=0)
    img_data = np.append(img_data, img, axis=0)


img_data = img_data[1:len]


#model = Net()

