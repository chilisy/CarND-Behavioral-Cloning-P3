import pandas as pd
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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


def read_all_images(image_names, angle, batch_size=32, flip=False, clr = 'c'):

    img_data = np.zeros((32, 32, 3))
    img_data = np.expand_dims(img_data, axis=0)

    start = 0
    end = start + batch_size
    n = image_names.shape[0]
    while start < n:
        if end > n:
            end = n

        img_batch = center[start:end]
        img_batch_data = np.zeros((32, 32, 3))
        img_batch_data = np.expand_dims(img_batch_data, axis=0)

        for img_name in img_batch:
            # read image
            img = cv2.imread(folder_data + img_name)
            # crop image
            img = img[50:140,:,:]
            # resize since generated data is too big to process otherwise
            img = cv2.resize(img, (32, 32))
            # normalize
            img = img / 255. - 0.5
            img = np.expand_dims(img, axis=0)
            # flip image
            if flip:
                img_batch_data = np.append(img_batch_data, np.fliplr(img), axis=0)
            else:
                img_batch_data = np.append(img_batch_data, img, axis=0)

        img_batch_data = img_batch_data[1:len(img_batch_data)]
        img_data = np.append(img_data, img_batch_data, axis=0)

        print(img_data.shape[0]-1, image_names.shape[0])

        start += batch_size
        end += batch_size

    img_data = img_data[1:len(img_data)]
    if flip:
        label_out = angle * -1
    else:
        label_out = angle

    bias = 0.24
    if clr=='l':
        label_out += bias
    elif clr=='r':
        label_out -= bias

    return img_data, label_out


# main script

folder_data = 'data/training_data_merge_2/'

batch_size = 32
augment_left_right = False

# read csv file
steering, throttle, brake, speed, center, left, right = load_data_from_log(folder_data + 'driving_log.csv')

# read images and augment flipped images
cimg, clab = read_all_images(center, steering, batch_size=32, flip=False, clr='c')
cimg_rev, clab_rev = read_all_images(center, steering, batch_size=32, flip=True, clr='c')

# augment images left right
if augment_left_right:
    limg, llab = read_all_images(left, steering, batch_size=32, flip=False, clr='l')
    limg_rev, llab_rev = read_all_images(left, steering, batch_size=32, flip=True, clr='l')

    rimg, rlab = read_all_images(right, steering, batch_size=32, flip=False, clr='r')
    rimg_rev, rlab_rev = read_all_images(right, steering, batch_size=32, flip=True, clr='r')

#img = np.concatenate((cimg, cimg_rev, limg, limg_rev, rimg, rimg_rev), axis=0)
#lab = np.concatenate((clab, clab_rev, llab, llab_rev, rlab, rlab_rev), axis=0)


cdata = {'images': cimg, 'steering': clab}
cdata_rev = {'images': cimg_rev, 'steering': clab_rev}
if augment_left_right:
    ldata = {'images': limg, 'steering': llab}
    ldata_rev = {'images': limg_rev, 'steering': llab_rev}
    rdata = {'images': rimg, 'steering': rlab}
    rdata_rev = {'images': rimg_rev, 'steering': rlab_rev}


with open('center_training_data.p', mode='wb') as f:
    pickle.dump(cdata, f)

with open('center_rev_training_data.p', mode='wb') as f:
    pickle.dump(cdata_rev, f)

if augment_left_right:
    with open('left_training_data.p', mode='wb') as f:
        pickle.dump(ldata, f)

    with open('right_training_data.p', mode='wb') as f:
        pickle.dump(rdata, f)

    with open('left_rev_training_data.p', mode='wb') as f:
        pickle.dump(ldata_rev, f)

    with open('right_rev_training_data.p', mode='wb') as f:
        pickle.dump(rdata_rev, f)


