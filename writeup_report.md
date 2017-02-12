#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center_2017_02_12_10_51_07_581.jpg "Center Lane Driving"
[image3]: ./examples/left_2017_02_12_21_04_53_780.jpg "Recovery Image"
[image4]: ./examples/center_2017_02_12_21_04_53_780.jpg "Recovery Image"
[image5]: ./examples/right_2017_02_12_21_04_53_780.jpg "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* preprocess_training_data.py read all image files and augment new images
* model2.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model2.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The preprocess_training_data.py contains the code for reading images and augment new images. 

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model2.py lines 48-61) 

The model includes ELU layers to introduce nonlinearity (model2.py line 53), and the data is normalized at preprocessing (preprocess_training_data line 52). 

####2. Attempts to reduce overfitting in the model

The model contains l2-regulization in order to reduce overfitting (model2.py lines 50). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 137). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model2.py line 79).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and a fast lap including cut corners.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The first approach I've chosen was using transfer learning with inception model for image processing and fully connected layer to generate output. Unfortunately, this approach did not work well, although I've achieved training loss of 0.02, the car did not stay on track. I've tried to trim parameters, but it didn't help and it was also not a issue of overfitting since I've added dropout layers. After lots of trying, I decided to start a new approach.

The new approach was to use a convolution neural network model similar to the nvidia paper. I thought this model might be appropriate because it was exactly the same task.

To combat the overfitting, I added l2-regulization from the start and overfitting was not a issue. 

This approach works better since I can reduce training loss to 0.006 and the car stays on track and could drive around corner.

There were a few spots where the vehicle fell off the track, especially after the bridge. To improve the driving behavior in these cases, I've added more training data of this corner but it did not help.


####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 

- convolutional layer, 3 1x1 filters to transform color space
- convolutional layer, 32 3x3 filters, l2 regulization
- convolutional layer, 32 3x3 filters
- activation, ELU
- convolutional layer, 64 3x3 filters
- convolutional layer, 64 3x3 filters
- activation, ELU
- convolutional layer, 128 3x3 filters
- convolutional layer, 128 3x3 filters
- maxpooling, 2x2
- activation, ELU
- flatten
- fully connected, 1600 outputs
- activation, ELU
- fully connected, 800 outputs
- fully connected, 16 outputs
- output-layer

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to center. These images show what a recovery looks like starting from the right side:

![alt text][image3]
![alt text][image4]
![alt text][image5]


After the collection process, I had X number of data points. I then preprocessed this data by augmenting them. For left and right images, I've added a bias of 0.24 to steering angle. I also generate new images by flipping them around, so the effect of this track, which contains more left than right corners, would be compensated.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3-4 as evidenced by 0.005. I used an adam optimizer so that manually training the learning rate wasn't necessary.