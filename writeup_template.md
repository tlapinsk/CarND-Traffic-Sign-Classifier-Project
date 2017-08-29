#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

Here is a link to my [project code](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Basic summary of the data set

I used the Python and Numpy to calculate summary statistics of the traffic signs data set:

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Shape of traffic sign image = (32, 32, 3)
Number of unique classes/labels = 43

####2. Include an exploratory visualization of the dataset.

Below is an explaratory visualization of the data set. First, 5 random images of the data set:

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/1.png "1")

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/22.png "22")

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/25.png "25")

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/28.png "28")

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/8.png "8")

The following includes 3 histograms showing the total image count for each class (training, validation, and test histograms are included):

Training Histogram

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/Training.png?raw=true "Training Histogram")

Validation Histogram

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/Validation.png?raw=true "Validation Histogram")

Test Histogram

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/Test.png?raw=true "Test Histogram")


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I first converted the images to grayscale for simplicity and data reduction. Simplicity because there is only one color channel, which thus translates to data reduction as well. Grayscale allows the neural network to only process 1/3 the amount of data compared to RBG.

[This stackoverflow](https://stackoverflow.com/questions/20473352/is-conversion-to-gray-scale-a-necessary-step-in-image-preprocessing) sums up my thoughts pretty well for why I converted to grayscale.

I then normalized the image data, so that the inputs were all within a comparable range. I believe Stanford's CS231n sums up data pre-processing techniques very well. Check it out [here](https://cs231n.github.io/neural-networks-2/#datapre).

Check out the before and after of my pre-processing below:

Original

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/Original.png "Original")

Normalized + Grayscale

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/NormalizedandGray.png "Normalized + Grayscale")

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 5x5x16 	|
| Flatten					| Outputs 400												|
| Fully connected		| Outputs 120        									|
| RELU					|												|
| Fully connected		| Outputs 84        									|
| RELU					|												|
| Fully connected		| Outputs 43        									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a slightly modified version of the LeNet architecture. You can find the original paper [here](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf). 

The sole modifications were to the batch size and total classes in the final output of the neural network. I decreased the batch size to get allow the neural network to take smaller step sizes during gradient descent. While this slows the learning process due to smaller step sizes, it allowes the network to converge closer to the optimum as you near it.

During my initial training of the neural network, I played around with the batch size, number of epochs, and learning rate quite a bit. I found that the original hyperparamers yielded the best results - with only batch size helping.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
Train Set Accuracy = 100%
Validation Set Accuracy = 96.3%
Test Set Accuracy = 92.8%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I chose the LeCun Architecture because it was very easy to get up and running. The only changes that needed to be made to the architecture were in the total number of classes output from the final fully connected layer.
* What were some problems with the initial architecture?
The initial architecture perpetually got stuck around 94% validation set accuracy when I first started training the model. 
* How was the architecture adjusted and why was it adjusted? 
I began testing out changes to the hypterparameters. This included changing the number of epochs (200, 150, etc.), batch size (64), and learning rate (both higher and lower). While these changes yielded different results, the only one that increased the validition set accuracy was dropping the batch size to 64. See above for why I believe this increased the validation set accuracy.

As a side note, increasing the number of epochs did not help push past any local minima that I hit. Also, increasing the learning rate allowed the net to descend faster, but "bounced" around and never got past local minima as well. Lowering the learning rate slowed down the descent time and also got stuck at local minima.

Moving on, I realized part way through my training that I wasn't converting my validation set to grayscale (oops!). This pushed the validation set accuracy to its peak with the LeCun architecture (96.3%).

I never encountered an overfitting or underfitting in my model. For the most part, I consistently got ~94-96% validation set accuracy, which is decent for the base LeCun Architecture.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
A convolution layer has become a staple for any computer vision or object recognition problem. I merely am following and replicating similar results to some of the best research that has been done in the field so far. In fact, my model could be significantly improved upon. More on that later...

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Train Set Accuracy = 100%
Validation Set Accuracy = 96.3%
Test Set Accuracy = 92.8%

The accuracy for each set above indicate that the model is working fairly well - with only ~4% point differences between each set of data. This is fairly standard (based on a quick Google search) and doesn't indicate overfitting. If the results were much farther apart, I would have scrapped the LeCun Architecture for something different.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](image4) ![alt text](image4) ![alt text](image4)
![alt text](image4) ![alt text](image4)

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


