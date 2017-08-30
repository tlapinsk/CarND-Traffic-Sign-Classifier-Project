# Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

## Writeup / README

Here is a link to my [project code](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

**Basic summary of the data set**

I used the Python and NumPy to calculate summary statistics of the traffic signs data set:

Number of training examples = 34799

Number of validation examples = 4410

Number of testing examples = 12630

Shape of traffic sign image = (32, 32, 3)

Number of unique classes/labels = 43

**Exploratory visualization of the data set**

Below is an explaratory visualization of the data set. First, 5 random images of the data set:

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/1.png "1")

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/22.png "22")

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/25.png "25")

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/28.png "28")

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/8.png "8")

The following histograms show the total image count for each class (training, validation, and test histograms are included):

Training Histogram

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/Training.png?raw=true "Training Histogram")

Validation Histogram

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/Validation.png?raw=true "Validation Histogram")

Test Histogram

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/Test.png?raw=true "Test Histogram")

Resources for exploratory visualization:
- [Matplotlib histogram documentation](https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.hist.html)
- [More Matplotlib documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist)
- [Creating a histogram with Seaborn](https://discussions.udacity.com/t/a-bit-of-help-with-matplotlib-please/224701/8?u=tim.lapinskas)
- [Seaborn histogram documentation](https://seaborn.pydata.org/tutorial/distributions.html)

### Design and Test a Model Architecture

**Pre-processing**

I first converted the images to grayscale for simplicity and data reduction. Simplicity because there is only one color channel, which thus translates to data reduction as well. Grayscale allows the neural network to only process 1/3 the amount of data compared to RBG.

[This stackoverflow](https://stackoverflow.com/questions/20473352/is-conversion-to-gray-scale-a-necessary-step-in-image-preprocessing) sums up my thoughts pretty well for why I converted to grayscale.

I then normalized the image data, so that the inputs were within a comparable range. I believe Stanford's CS231n sums up data pre-processing techniques very well. Check it out [here](https://cs231n.github.io/neural-networks-2/#datapre).

Before and after images below:

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/Original.png "Original")
![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/NormalizedandGray.png "Normalized + Grayscale")

Resources for pre-processing:
- [Data normalization forum post](https://discussions.udacity.com/t/normalizing-the-data/333593)

**Final model architecture**

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten					| Outputs 400												|
| Fully connected		| Outputs 120        									|
| RELU					|												|
| Fully connected		| Outputs 84        									|
| RELU					|												|
| Fully connected		| Outputs 43        									|
 
**Hypterparameter tuning**

To train the model, I used a slightly modified version of the LeNet architecture. You can find the original paper [here](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf). 

The only modifications were to the batch size and class output on the last layer of the network. I decreased the batch size to allow the neural network to take smaller steps during gradient descent. While this slows the learning process due to smaller step sizes, it lets the network to converge closer to the optimum as you near it.

During my initial training of the neural network, I played around with the batch size, number of epochs, and learning rate quite a bit. I found that the original hyperparamers yielded the best results - with only batch size helping.

**Solution discussion**

Code that calculated the accuracy of each data set:
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    num_examples = len(X_train)
	    
	    print("Training...")
	    print()
	    for i in range(EPOCHS):
	        X_train, y_train = shuffle(X_train, y_train)
	        X_valid, y_valid = shuffle(X_valid, y_valid)
	        X_test, y_test = shuffle(X_test, y_test)
	        for offset in range(0, num_examples, BATCH_SIZE):
	            end = offset + BATCH_SIZE
	            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
	            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
	        
	        train_accuracy = evaluate(X_train, y_train)
	        validation_accuracy = evaluate(X_valid, y_valid)
	        test_accuracy = evaluate(X_test, y_test)
	        print("EPOCH {} ...".format(i+1))
	        print("Train Accuracy = {:.3f}".format(train_accuracy))
	        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
	        print("Test Accuracy = {:.3f}".format(test_accuracy))
	        print()
	        
	    saver.save(sess, './lenet')
	    print("Model saved")

Final results:
Train Set Accuracy = 100%
Validation Set Accuracy = 96.3%
Test Set Accuracy = 92.8%

* What was the first architecture that was tried and why was it chosen?
I chose the LeNet architecture because it was very easy to get up and running. The only change that needed to be made was in the last fully connected layer.
* What were some problems with the initial architecture?
The initial architecture perpetually got stuck around 94% validation set accuracy when I first started training the model. 
* How was the architecture adjusted and why was it adjusted? 
I began testing out changes to the hypterparameters. This included changing the number of epochs (200, 150, etc.), batch size (64), and learning rate (both higher and lower). While these changes yielded different results, the only one that increased the validation set accuracy was batch size. See above for why I believe this increased the validation set accuracy.

As a side note, increasing the number of epochs did not help push past any local minima that I hit. Also, increasing the learning rate allowed the net to descend faster, but "bounced" around and never got past local minima. Lowering the learning rate slowed down the descent time and also got stuck at local minima.

Moving on, I realized part way through my training that I wasn't converting my validation set to grayscale (oops!). This pushed the validation set accuracy to its peak with the LeCun architecture (96.3%).

I never encountered any overfitting or underfitting in my model. For the most part, I consistently hit ~94-96% validation set accuracy, which is decent for the base LeNet architecture.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
A convolution layer has become a staple for any computer vision or image recognition problem. I merely am following and replicating similar results based on some of the best research that has been done in the field so far. In fact, my model could be significantly improved upon. More on that later...

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Train Set Accuracy = 100%
Validation Set Accuracy = 96.3%
Test Set Accuracy = 92.8%

The accuracy for each set above indicate that the model is working fairly well - with only ~4% point differences between each set of data. This is fairly standard (based on a quick Google search) and doesn't indicate overfitting. If the results were much farther apart, I would have scrapped the LeNet architecture for something different.

### Test a Model on New Images

**Five German traffic signs**

Here are five German traffic signs that I found on the web:

![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/web1.png "Web Pic #1") 
![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/web2.png "Web Pic #2") 
![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/web3.png "Web Pic #3")
![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/web4.png "Web Pic #4") 
![alt text](https://github.com/tlapinsk/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/web5.png "Web Pic #5")

I believe these pictures will actually be fairly easy to classify since the signs take up the majority of each picture and are fairly square images.

**Model's predictions**

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      		| #11 Right-of-way at the next intersection  									| 
| Speed limit (60km/h)     			| #3 Speed limit (60km/h)										|
| Speed limit (30km/h)					| #1 Speed limit (30km/h)										|
| Stop	      		| #14 Stop				 				|
| Road work		| #25 Road work     							|

The model was able to correctly guess 5 out of 5 traffic signs, which gives an accuracy of 100%. This compares similarly to the accuracy of the training, validation, and test sets (100%, 96.3%, and 92.8% respectively)

**Model certainty & softmax**

For the first image, the model is 100% sure that it is a right-of-way at the next intersection sign, and the image does contain this sign. The top fix soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Right-of-way at the next intersection  									| 
| 7.85e-26     				| Pedestrians 										|
| 1.78e-29					| Priority road									|
| 1.18e-29	      			| Roundabout mandatory				 				|
| 5.07e-30				    | Double curve      							|

For the second image, the model is 100% sure that it is a speed limit (60km/h) sign, and the image does contain this sign. The top fix soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Speed limit (60km/h) 									| 
| 2.95e-13     				| Speed limit (50km/h)										|
| 1.33e-16					| Slippery road									|
| 4.18e-20	      			| Speed limit (80km/h)				 				|
| 1.51e-22				    | End of speed limit (80km/h)      							|

For the third image, the model is 100% sure that it is a speed limit (30km/h) sign, and the image does contain this sign. The top fix soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99e-1         			| Speed limit (30km/h)	 									| 
| 1.04e-05     				| Speed limit (50km/h)										|
| 4.03e-17					| Speed limit (60km/h)								|
| 3.54e-20	      			| Speed limit (80km/h)				 				|
| 1.50e-24				    | Stop      							|

For the fourth image, the model is 100% sure that it is a stop sign, and the image does contain this sign. The top fix soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Stop 									| 
| 4.62e-05     				| Speed limit (80km/h) 										|
| 1.96e-07					| No passing for vehicles over 3.5 metric tons									|
| 3.68e-08	      			| Turn left ahead				 				|
| 3.57e-08				    | Speed limit (50km/h)      							|

For the fifth image, the model is 100% sure that it is a road work sign, and the image does contain this sign. The top fix soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Road work  									| 
| 2.89e-21     				| Right-of-way at the next intersection 										|
| 8.51e-22					| Beware of ice/snow									|
| 6.75e-29	      			| Yield				 				|
| 6.69e-29				    | Turn left ahead      							|

Resources for softmax:
- [Forum post about softmax errors](https://discussions.udacity.com/t/softmax-probability-issue/242414)
- [Forum post about softmax results](https://discussions.udacity.com/t/low-softmax-probability-in-images-from-web/248462/22?u=tim.lapinskas)

### Improving the model further

Due to limited time on this project, I did not pursue going above and beyond the basic requirements in the rubric. I would like to add a few notes about how my model could be further improved though - see below:

**Pick a new architecture**
The LeNet architecture seems to have hit its max validation accuracy around 96%. Although, please provide feedback if you have seen it perform above 96% - I am definitely curious!

Other architectures that I believe would be more successful are below:
- [VGG Net](https://arxiv.org/pdf/1409.1556v6.pdf)
- [GoogLenet](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)
- [GAN](https://arxiv.org/pdf/1406.2661v1.pdf)

These are only a few of the models that I believe would allow close to 100% accuracy for all sets of data. It would be fun to revisit this project and attempt implementing different networks.

**Data augmentation** 
As noted in the project rubric, data augmentation (flips, zoom, translation, and color pertubation) would allow for the network to train on more data. As we all know, the more data the better the results.

**Visualize layers of the network**
This one is really interesting to me. Having seen how other computer scientists have visualized layers of their neural networks, it would be cool to see how LeNet visualized the data in each layer. Albeit, it wouldn't increase performance of the network.