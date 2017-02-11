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


folder_data = 'data/'
test_data_file = 'training_data.p'

steering, throttle, brake, speed, center, left, right = load_data_from_log(folder_data + 'driving_log.csv')

test_img = cv2.imread(folder_data+center[0])

cimg_data = np.zeros_like(test_img)
cimg_data = np.expand_dims(cimg_data, axis=0)
#limg_data = np.zeros_like(test_img)
#limg_data = np.expand_dims(limg_data, axis=0)
#rimg_data = np.zeros_like(test_img)
#rimg_data = np.expand_dims(rimg_data, axis=0)

for cimg_name, limg_name, rimg_name in zip(center, left, right):
    cimg = cv2.imread(folder_data+cimg_name)
    cimg = np.expand_dims(cimg, axis=0)
    cimg_data = np.append(cimg_data, cimg, axis=0)

    #limg = cv2.imread(folder_data + limg_name)
    #limg = np.expand_dims(limg, axis=0)
    #limg_data = np.append(limg_data, limg, axis=0)

    #rimg = cv2.imread(folder_data + rimg_name)
    #rimg = np.expand_dims(rimg, axis=0)
    #rimg_data = np.append(rimg_data, rimg, axis=0)


cimg_data = cimg_data[1:len(cimg_data)]
#limg_data = limg_data[1:len(limg_data)]
#rimg_data = rimg_data[1:len(rimg_data)]

test_data_file_obj = open(test_data_file, 'wb')

#stored_data = {'center': cimg_data, 'left': limg_data, 'right': rimg_data, 'steering': steering}
stored_data = {'center': cimg_data, 'steering': steering}

pickle.dump(stored_data, test_data_file_obj)

# img_data.shape
# steering.shape

