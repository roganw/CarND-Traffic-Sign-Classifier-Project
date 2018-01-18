# **Traffic Sign Recognition** 

[//]: # (Image References)

[image1]: ./examples/pie.jpg  
[image2]: ./examples/bar.jpg 

[image3]: ./test/00298.ppm  
[image4]: ./test/00818.ppm  
[image5]: ./test/00963.ppm  
[image6]: ./test/01057.ppm  
[image7]: ./test/01606.ppm
[image8]: ./test/01666.ppm
[image9]: ./test/01918.ppm
[image10]: ./test/04202.ppm

[image11]: ./challenge/front.jpeg
[image12]: ./challenge/side.jpeg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/roganw/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

1.1 Generate validation data  
I used `train_test_split()` method to split 20% validation data from training data, and saved as `./data/valid.p`
```python
train_X, valid_X, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
```

1.2 Data Set Summary  
I used the python/numpy to calculate summary statistics of the traffic signs data set:

* The size of training set is 31367
* The size of the validation set is 7842
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

This is a pie chart showing the percentage of training set, validation set and test set.

![alt text][image1]

This is a bar chart showing the quantity of training set, validation set and test set on each class.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 

I used `shuffle()` method to reorder the training set in the pre-process section.
```python
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)
```


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3    	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 14x14x6 	|
| Convolution 5x5x6	    | 1x1 stride, VALID padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 5x5x16 	|
| Flattern      		| Input = 5x5x16, Output = 400		    		|
| Fully connected		| Input = 400, Output = 120						|
| RELU					|												|
| Fully connected		| Input=84, Output = n_classes (43)     									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used `AdamOptimizer` of TensorFlow as the optimizer:
```python
BATCH_SIZE = 128
EPOCHS = 30
rate = 0.001
# ...
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
```

The model was trained by feed the training set to the model, and looping executed according to the epochs, batch size parameters.
```python
for i in range(EPOCHS):
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        sess.run(training_operation, feed_dict={x: batch_x, y:batch_y})
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.964
* test set accuracy of 0.895

The iterative approach I chosen was referred from LeNet. 
I increased `EPOCHS` to 30 to advance the accuracy.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 8 German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10]

And there are 2 more picture with mosaics download from the Internet, which are publicly known easy to misjudged to `Speed imit 45`：
![alt text][image11] ![alt text][image12]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Predict the Sign Type for Each Image:
```python
predictions = tf.nn.softmax(logits)
labels = tf.argmax(predictions, axis=1)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    predict_labels = sess.run(labels, feed_dict={x: X_test_new, y:y_test_new})
    print(predict_labels)
```
Here are the results of the prediction:

| Class     | Image			        |     Prediction	        					| 
|:---------:|:---------------------:|:---------------------------------------------:| 
| 35        | Ahead only      		| Ahead only   									| 
| 1         | Speed limit (30km/h)	| Speed limit (30km/h) 	     					|
| 14        | Stop					| Stop											|
| 33        | Turn right ahead	    | Turn right ahead					        	|
| 18        | General caution		| General caution	                            |
| 16        | Vehicles over 3.5 metric tons prohibited | Vehicles over 3.5 metric tons prohibited |
| 25        | Road work		        | Road work     						    	|
| 13        | Yield		            | Yield    							            |
| ----      | -----                 | -----      					                |
| 14        | Stop			        | Stop    					                    |
| 14        | Stop			        | Stop   							            |


The model was able to correctly guess all of the 10 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model:
```python
predictions = tf.nn.softmax(logits)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    top_keys = sess.run(tf.nn.top_k(predictions, k=3), feed_dict={x: X_test_new, y: y_test_new})
    print(top_keys)
```
Max probability for every image:

| Class     | Probability   |     Prediction	        					| 
|:---------:|:-------------:|:---------------------------------------------:| 
| 35        | 1             | Ahead only   									| 
| 1         | 1	            | Speed limit (30km/h) 	     					|
| 14        | 1				| Stop											|
| 33        | 1             | Turn right ahead					        	|
| 18        | 1	            | General caution	                            |
| 16        | 1             | Vehicles over 3.5 metric tons prohibited |
| 25        | 1	            | Road work     						    	|
| 13        | 1	            | Yield    							            |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


