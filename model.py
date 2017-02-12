from keras.layers.core import Dropout
from keras.layers import Input, Flatten, Dense
from keras.layers.advanced_activations import ELU
from keras.models import Model
from keras.regularizers import l2

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', 'inception_drive_bottleneck_features_train.p',
                    "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', 'inception_drive_bottleneck_features_validation.p',
                    "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 64, "The batch size.")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)
    with open('steering_angle_min_max.p', mode='rb') as f:
        angle_min_max = pickle.load(f)
    amin = angle_min_max['min']
    amax = angle_min_max['max']

    X_train = train_data['features']
    y_train = train_data['labels']
    y_train = (y_train-amin)/(amax-amin)-0.5
    X_val = validation_data['features']
    y_val = validation_data['labels']
    y_val = (y_val-amin)/(amax-amin)-0.5

    return X_train, y_train, X_val, y_val


def net(input_shape):

    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    x = Dropout(0.8)(x)
    x = Dense(2048)(x)
    x = ELU(alpha=1.0)(x)
    x = Dropout(0.9)(x)
    x = Dense(512)(x)
    x = ELU(alpha=1.0)(x)
    x = Dense(10)(x)
    x = ELU(alpha=1.0)(x)
    x = Dense(1)(x)
    #x = ELU(alpha=1.0)(x)
    model = Model(inp, x)
    model.compile(optimizer='adam', loss='mse')

    return model


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # define model
    input_shape = X_train.shape[1:]
    model = net(input_shape)

    # train model
    model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size,
              validation_data=(X_val, y_val), shuffle=True)

    model.save('model.h5')
# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

