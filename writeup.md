# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./training_data.png "Visualization"
[image2]: ./validation_data.png "Visualization"
[image3]: ./test_data.png "Visualization"
[image4]: ./examples/grayscale.jpg "Grayscaling"
[image5]: ./examples/random_noise.jpg "Random Noise"
[image6]: ./my_images/2_speedlimit50.jpeg "Traffic Sign 1"
[image7]: ./my_images/4_speedlimit70.jpeg "Traffic Sign 2"
[image8]: ./my_images/12_priorityroad.jpeg "Traffic Sign 3"
[image9]: ./my_images/14_stop.jpeg "Traffic Sign 4"
[image10]: ./my_images/14_stop2.jpeg "Traffic Sign 5"
[image11]: ./my_images/18_GeneralCaution.jpeg "Traffic Sign 6"
[image12]: ./my_images/27_Pedestrians.jpeg "Traffic Sign 7"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/rkopec91/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy's shape attribute to find the number of traing, validation, and testing examples.  I also used shape to find the input shape of thes images.  To find the number of classes, I used the set function to turn the labels into a set, then I took the length of that.  The following are my findings:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1] ![alt text][image2] ![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In order to preprocess the images, I converted them to gray scale.  Color is important when it comes to sign detection, but for this project, the signs that are shown, can be distinguished by their shape and what other details are on it.  None of the signs have the same shape/detail but different colors.  So, we can just ignore color for this project.

The next thing that is done is I normalized the image.  The input image was from 0 to 255 but I normalized it to 0 to 1.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         	|     Description				| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x1 RGB image  				| 
| Convolution 4x4     	| 1x1 stride, same padding, outputs 28x28x12 	|
| RELU			|					        |
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 		|
| Convolution 4x4     	| 1x1 stride, same padding, outputs 10x10x25 	|
| RELU			|					        |
| Max pooling	      	| 2x2 stride,  outputs 5x5x25 	        	|
| Flatten               |                                               |
| Fully connected	| outputs 300                                   |
| RELU                  |                                               |
| Fully connected	| outputs 100                                   |
| RELU                  |                                               |
| Output		| outputs 43					|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I decided to run the model through 100 epochs with a batch size of 128.  I had a keep probability of 0.8.  So this would have 80 percent chance of keeping the connection between the nodes.  The learning rate that I used was 0.001.  I didn't want to go to small or too large of a learning rate or it would take too long to converge or it would have kept overshooting the optimal weights.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.954 
* test set accuracy of 0.943

I chose to go with a lenet architecture.  I chose this because it would give good results.  It had been proven to output great accuracy for the mnist dataset so it should work well on this dataset.  These were all high accuracies.  They all (training, validation, and test sets) scored over 94 percent accuracy ratings.  This proves that the model performs well.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10] ![alt text][image11]
![alt text][image12]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			|     Prediction	 			| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      	| Stop sign                                     | 
| U-turn     		| U-turn                                        |
| Yield		        | Yield						|
| 100 km/h	      	| Bumpy Road					|
| Slippery Road		| Slippery Road      				|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| .60         		| Stop sign                                     | 
| .20     		| U-turn                                        |
| .05			| Yield					        |
| .04	      		| Bumpy Road					|
| .01		        | Slippery Road      				|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


