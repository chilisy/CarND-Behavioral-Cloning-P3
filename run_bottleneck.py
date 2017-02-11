from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, AveragePooling2D
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.datasets import cifar10
import pickle, cv2
import tensorflow as tf
import keras.backend as K
import pandas as pd
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('driving_log_file', 'data/driving_log.csv', "Make bottleneck features")
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
        folder = 'data/'

        while True:

            img_batch = data[start:end]
            images = np.zeros((299, 299, 3))
            images = np.expand_dims(images, axis=0)
            for img_name in img_batch:
                center_image = cv2.imread(folder + img_name)
                center_image = cv2.resize(center_image, (299, 299))
                center_image = center_image/255.-0.5
                center_image = np.expand_dims(center_image, axis=0)
                images = np.append(images, center_image, axis=0)

            # trim image to only see section with road
            images = images[1:len(images)]

            #X_batch = session.run(resize_op, {img_placeholder: images})
            #X_batch = preprocess_input(X_batch)
            X_batch = images
            y_batch = labels[start:end]
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
    x = AveragePooling2D((8, 8), strides=(8, 8))(x)
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

        Img_train, Img_val, ang_train, ang_val = train_test_split(center, steering, test_size=0.2)

        print('Bottleneck training')
        train_gen = gen(sess, Img_train, ang_train, batch_size)
        bottleneck_features_train = model.predict_generator(train_gen(), Img_train.shape[0])
        data = {'features': bottleneck_features_train, 'labels': ang_train}
        pickle.dump(data, open(train_output_file, 'wb'))

        print('Bottleneck validation')
        val_gen = gen(sess, Img_val, ang_val, batch_size)
        bottleneck_features_validation = model.predict_generator(val_gen(), Img_val.shape[0])
        data = {'features': bottleneck_features_validation, 'labels': ang_val}
        pickle.dump(data, open(validation_output_file, 'wb'))

if __name__ == '__main__':
    tf.app.run()
