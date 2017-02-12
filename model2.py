from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.models import Sequential

from sklearn.model_selection import train_test_split
import keras.backend as K

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('driving_log_file', 'data/training_data_merge/driving_log.csv', "Make bottleneck features")
flags.DEFINE_integer('epochs', 4, "The number of epochs.")
flags.DEFINE_integer('batch_size', 16, "The batch size.")


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
    index = np.array([speed[i]<35 and speed[i]>30 and throttle[i]>0.95 for i in range(len(speed))])

    return steering[index], throttle[index], brake[index], speed[index], center[index], left[index], right[index]


def net():

    model = Sequential()
    # Convolutional Layers for image processing
    model.add(Convolution2D(3, 1, 1, input_shape=(32, 32, 3)))

    model.add(Convolution2D(32, 3, 3, W_regularizer=l2(0.0001), b_regularizer=l2(0.0001)))
    model.add(Convolution2D(32, 3, 3))

    model.add(ELU(alpha=1.0))
    #model.add(Dropout(0.9))

    model.add(Convolution2D(64, 3, 3))
    model.add(Convolution2D(64, 3, 3))
    model.add(MaxPooling2D((2, 2)))
    model.add(ELU(alpha=1.0))

    model.add(Convolution2D(128, 3, 3))
    model.add(Convolution2D(128, 3, 3))
    model.add(MaxPooling2D((2, 2)))
    model.add(ELU(alpha=1.0))

    # fully connected layers
    model.add(Flatten())

    model.add(Dense(1600))
    model.add(ELU(alpha=1.0))

    model.add(Dense(800))
    model.add(ELU(alpha=1.0))

    model.add(Dense(16))
    #model.add(ELU(alpha=1.0))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model


def main(_):
    with open(FLAGS.driving_log_file) as f:
        steering, throttle, brake, speed, center, left, right = load_data_from_log(f)


    # max_steering = max(steering)
    # min_steering = min(steering)
    # angle_min_max = {'min': min_steering, 'max': max_steering}
    #
    # with open('steering_angle_min_max.p', mode='wb') as f:
    #     pickle.dump(angle_min_max, f)

    with open('center_training_data.p', mode='rb') as f:
        data = pickle.load(f)
        cimg = data['images']
        cang = data['steering']

    with open('center_rev_training_data.p', mode='rb') as f:
        data = pickle.load(f)
        cimg_rev = data['images']
        cang_rev = data['steering']

    # with open('left_training_data.p', mode='rb') as f:
    #     data = pickle.load(f)
    #     limg = data['images']
    #     lang = data['steering']
    #
    # with open('left_rev_training_data.p', mode='rb') as f:
    #     data = pickle.load(f)
    #     limg_rev = data['images']
    #     lang_rev = data['steering']
    #
    # with open('right_training_data.p', mode='rb') as f:
    #     data = pickle.load(f)
    #     rimg = data['images']
    #     rang = data['steering']
    #
    # with open('right_rev_training_data.p', mode='rb') as f:
    #     data = pickle.load(f)
    #     rimg_rev = data['images']
    #     rang_rev = data['steering']


    #img = np.concatenate((cimg, cimg_rev, limg, limg_rev, rimg, rimg_rev), axis=0)
    #ang = np.concatenate((cang, cang_rev, lang, lang_rev, rang, rang_rev), axis=0)

    img = np.concatenate((cimg, cimg_rev), axis=0)
    ang = np.concatenate((cang, cang_rev), axis=0)

    with tf.Session() as sess:
        K.set_session(sess)
        K.set_learning_phase(1)

        X_train, X_val, y_train, y_val = train_test_split(img, ang, test_size=0.2)

        # define model
        model = net()

        # train model
        model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size,
                  validation_data=(X_val, y_val), shuffle=True)

        model.save('model.h5')

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()