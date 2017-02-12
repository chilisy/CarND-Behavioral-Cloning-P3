from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, AveragePooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.datasets import cifar10
import pickle, cv2
import tensorflow as tf
import keras.backend as K
import pandas as pd
import numpy as np
import math

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('driving_log_file', 'data/training_data_merge/driving_log.csv', "Make bottleneck features")
flags.DEFINE_integer('batch_size', 32, 'The batch size for the generator')

batch_size = FLAGS.batch_size


h, w, ch = 299, 299, 3

img_placeholder = tf.placeholder("uint8", (None, 160, 320, 3))
resize_op = tf.image.resize_images(img_placeholder, (h, w), method=0)


def gen(session, data, labels, batch_size):
    def _f():
        start = 0
        end = start + batch_size
        n = data.shape[0]
        folder = 'data/training_data_merge/'

        while True:

            img_batch = data[start:end]
            images = np.zeros((299, 299, 3))
            images = np.expand_dims(images, axis=0)
            images_reversed = np.zeros((299, 299, 3))
            images_reversed = np.expand_dims(images_reversed, axis=0)
            for img_name in img_batch:
                img = cv2.imread(folder + img_name)
                #img = img[math.floor(img.shape[0] / 5):img.shape[0] - 25, 0:img.shape[1]]
                img = cv2.resize(img, (299, 299))
                img = img/255.-0.5
                img = np.expand_dims(img, axis=0)
                images = np.append(images, img, axis=0)
                images_reversed = np.append(images_reversed, np.fliplr(img), axis=0)

            images = images[1:len(images)]
            images_reversed = images_reversed[1:len(images_reversed)]

            X_batch = np.append(images, images_reversed, axis=0)
            y_batch = np.append(labels[start:end], labels[start:end]*-1)
            start += batch_size
            end += batch_size
            if start >= n:
                start = 0
                end = batch_size

            print(start, end)
            yield (X_batch, y_batch)

    return _f


def create_model():
    input_tensor = Input(shape=(h, w, ch))

    model = InceptionV3(input_tensor=input_tensor, include_top=False)
    x = model.output
    x = AveragePooling2D((1, 1), strides=(1, 1))(x)
    model = Model(model.input, x)

    return model


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
    index = np.array([speed[i]<35 and speed[i]>30 and throttle[i]>0.95 and steering[i]>-1 and steering[i]<1
                      for i in range(len(speed))])

    return steering[index], throttle[index], brake[index], speed[index], center[index], left[index], right[index]


def generate_training_images(left, right, center, steering):
    bias = 0.25
    steering_left = steering + bias
    steering_right = steering - bias

    images = np.append(center, left)
    steering_angles = np.append(steering, steering_left)
    images = np.append(images, right)
    steering_angles = np.append(steering_angles, steering_right)

    return images, steering_angles


def gen_label_data(labels, batch_size):
    start = 0
    end = start + batch_size
    n = labels.shape[0]
    label_out = []
    while start < n:
        if end > n:
            end = n

        y_batch = np.append(labels[start:end], labels[start:end] * -1)
        label_out = np.append(label_out, y_batch)
        start += batch_size
        end += batch_size

    return label_out


def main(_):

    with open(FLAGS.driving_log_file) as f:
        steering, throttle, brake, speed, center, left, right = load_data_from_log(f)

    max_steering = max(steering)
    min_steering = min(steering)
    angle_min_max = {'min': min_steering, 'max': max_steering}

    with open('steering_angle_min_max.p', mode='wb') as f:
        pickle.dump(angle_min_max, f)

    train_output_file = "{}_{}_{}.p".format('inception', 'drive', 'bottleneck_features_train')
    validation_output_file = "{}_{}_{}.p".format('inception', 'drive', 'bottleneck_features_validation')

    img_names, ste_angles = generate_training_images(left, right, center, steering)

    print("Resizing to", (w, h, ch))
    print("Saving to ...")
    print(train_output_file)
    print(validation_output_file)

    with tf.Session() as sess:
        K.set_session(sess)
        K.set_learning_phase(1)

        model = create_model()

        model.compile(optimizer='adam', loss='mse')
        model.save('pre_model.h5')

        Img_train, Img_val, ang_train, ang_val = train_test_split(img_names, ste_angles, test_size=0.2)

        ang_train = gen_label_data(ang_train, batch_size)
        ang_val = gen_label_data(ang_val, batch_size)

        print('Bottleneck training')
        train_gen = gen(sess, Img_train, ang_train, batch_size)
        bottleneck_features_train = model.predict_generator(train_gen(), Img_train.shape[0]*2)
        data = {'features': bottleneck_features_train, 'labels': ang_train}
        pickle.dump(data, open(train_output_file, 'wb'))

        print('Bottleneck validation')
        val_gen = gen(sess, Img_val, ang_val, batch_size)
        bottleneck_features_validation = model.predict_generator(val_gen(), Img_val.shape[0]*2)
        data = {'features': bottleneck_features_validation, 'labels': ang_val}
        pickle.dump(data, open(validation_output_file, 'wb'))

if __name__ == '__main__':
    tf.app.run()
