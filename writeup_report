**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Data analysis can be found on the first cell of the - [behavioralCloning.ipynb](https://github.com/sistelexis/behavioralCloning/blob/master/behavioralCloning.ipynb) file
Data processing can be seen on the following cells.
On that file, it can also be found the different steps of improving the model, from the first recomemndation, to the last model that completed the lap on the track one.

Visualization of 3 randomly chosen images

![visualization of 3 randomly chosen images](images/3images.png)

Crop action on those 3 images

![crop action on those 3 images](images/3images_cropped.png)

Flip action on those 3 images

![flip action on those 3 images](images/3images_flipped.png)

Brightness adjustment on those 3 images

![brightness adjustment on those 3 images](images/3images_brightness.png)

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following mandatory files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
I also added the following files to support the writeup_report.md file:
* behavioralCloning.ipynb

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

After testing the different models suggested by the videos (basic, LeNet and Nvidia models), I decided to go deeper with my analysis using the Nvidia model since it is a model known to be working for this type of projects.
After adding the normalization using a Keras lambda layer (to deal with value between -0.5 and 0.5 as suggested), I also used Cropping2D in order to get rid of the useless information from the picture (sky, trees, car). Using the cropping on the model allows to have that cropping same done once testing the model on the simulator. (model.py lines 91 - 108) 

With that model, I was able to analyze the recorded images from the simulator. Since getting valid images using the keyboard or even the mouse was getting quite difficult, I decided to use the provided dataset, and only get back to the simulator if I wouldn't manage to get things working.

####2. Attempts to reduce overfitting in the model

Although overfitting has never been obvious, I added dropouts to handle see the model reaction when the borders changed and when the same lane lines type was used in straights and curves.
The model contains dropout layers in order to reduce overfitting (model.py lines 95, 97, 99, 101, 103). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 110).

####4. Appropriate training data

After testing with just the center image it was clear that the system had not enough information to learn how to handle tight curves.
So as suggested, the most obvious data augmentation has been used:
* use left and right images and then adjust the steering by a value to simulate a center line recovery. The adjustment value as been decided based on the average steering from the dataset (around 0.15), as it can be seen on the ipynb provided. I tested with bigger values, but the car, although it managed to drive inside the authorized limits, had a "nervous" behavior, waving from one side of the road to the other.
* flip images to simulate more right curves in order also to level images that leads to both sides steering. Following the same logic, I only flipped images that had a more than 0.1 of absolute steering value, because by a big margin, most of the images were for going strait or very small steering.

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvidia model. I understood that this model would be appropriate because it is a proven one in that field.

From the start, I did not have a significant overfitting, but even then I preferred to add dropouts since it was not clear how the model handled some curves where the lane lines were the same as on the straights. I saw a small improvement on that side so I preferred to keep the dropouts. The value of 0.4 was the first attempt, and based on the good results, I decided to keep it.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I realized also that some of them were mainly caused by code that needed to be tweaked on the generator. I recommend to be very careful here, since small mistakes may lead to very erratic behaviors from the car. And the worst part is that they are sometimes very hard find.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 color image   		    			| 
| Normalization         |												|
| Cropping2D            | Remove 70 top and 25 bottom lines from images |
| Convolution2D 5x5    	| 2x2 stride, 24 layers                     	|
| RELU					|												|
| Droptout				| 0.4											|
| Convolution2D 5x5    	| 2x2 stride, 36 layers                     	|
| RELU					|												|
| Droptout				| 0.4											|
| Convolution2D 5x5    	| 2x2 stride, 48 layers                     	|
| RELU					|												|
| Droptout				| 0.4											|
| Convolution2D 5x5    	| 64 layers                     	            |
| RELU					|												|
| Droptout				| 0.4											|
| Convolution2D 5x5    	| 64 layers                                  	|
| RELU					|												|
| Droptout				| 0.4											|
| Flatten   	      	| 												|
| Fully connected		| outputs 100        							|
| Fully connected		| outputs 50        							|
| Fully connected		| outputs 10        							|
| Fully connected		| outputs 1         							|

####3. Creation of the Training Set & Training Process

Since getting valid images using the keyboard or even the mouse was getting quite difficult, I decided to use the provided dataset, and only get back to the simulator if I wouldn't manage to get things working. What at the end, did not happen.

I randomly shuffled the data set and put 20% of the initial data into a validation set. 

To augment the data set, as explained above, I flipped images and also used the left and right camera images

After the collection process, I would be able to get as much as 6 times the original dataset, but to avoid overfitting I preferred not to augment data using images with steering values under 0.1.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I started working with 3 epochs and was not getting a significant improvement from the second to the third epoch, so I never tried going further. I used an Adam optimizer so that manually training the learning rate wasn't necessary.

###Other information about the way I completed the project

Since running that model was a bit heavy for my laptop I setup a server with dual XEONs and 96G of RAM. Although I had not been able to use a GPU, the server was powerful enough to allow testing the model. As seen from others experience, it is a very good help to have a powerful hardware when we are searching in the "dark" how to get projects like this one to work.

Another point referenced by others on the recommended Paul Heraty guide, is to be very careful with the color set. Avoiding the dirt exits came to me only following this tip. If opening the images with cv2, we must convert them immediately from BGR to RGB in order to have it working with the drive.py
