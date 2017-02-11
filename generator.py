import pandas as pd
import pickle
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.applications import InceptionV3
from keras.layers.core import Lambda
from keras.models import load_model

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

    # use only frames where the car was at full speed
    index = np.array([speed[i] < 35 and speed[i] > 30 and throttle[i] > 0.95 for i in range(len(speed))])

    return steering[index], throttle[index], brake[index], speed[index], center[index], left[index], right[index]


def generator(img_names, steering_angle, folder, batch_size=32):
    num_samples = steering_angle.shape[0]
    while 1: # Loop forever so the generator never terminates
        shuffle(img_names, steering_angle)
        test_image = cv2.imread(folder + img_names[0])
        for offset in range(0, num_samples, batch_size):
            img_batch = img_names[offset:offset+batch_size]
            ang_batch = steering_angle[offset:offset+batch_size]
            images = np.zeros_like(test_image)
            images = np.expand_dims(images, axis=0)
            angles = np.array([0])
            for img_name, steer in zip(img_batch, ang_batch):

                center_image = cv2.imread(folder+img_name)
                center_angle = steer
                center_image = np.expand_dims(center_image, axis=0)
                #center_image = batch_samples['center']
                #center_angle = batch_samples['steering']
                images = np.append(images, center_image, axis=0)
                angles = np.append(angles, center_angle)

            # trim image to only see section with road
            X_train = images[1:len(images)]
            y_train = angles[1:len(angles)]
            yield shuffle(X_train, y_train)


folder_data = 'data/'
untrained_model_file = 'untrained_model.h5'
steering, throttle, brake, speed, center, left, right = load_data_from_log(folder_data + 'driving_log.csv')

data = {'image': center, 'angle': steering}
with open('traning_data.p', mode='wb') as f:
    pickle.dump(data, f)

Img_train, Img_val, ang_train, ang_val = train_test_split(center, steering, test_size=0.2)
# compile and train the model using the generator function
train_generator = generator(Img_train, ang_train, folder_data, batch_size=32)
validation_generator = generator(Img_val, ang_val, folder_data, batch_size=32)

ch, row, col = 3, 160, 320  # Trimmed image format

if os.path.isfile(untrained_model_file):
    model = load_model(untrained_model_file)
else:
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))
    model.add(InceptionV3(include_top=False, weights='imagenet'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.save(untrained_model_file)

train_gen = train_generator()
train_feature = model.predict_generator(train_generator, val_samples=len(ang_train))
validation_feature = model.predict_generator(validation_generator, val_samples=len(ang_val))

