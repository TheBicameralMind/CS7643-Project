 # Introduction / Background / Motivation

### What did you try to do? What problem did you try to solve? 

* Design lightweight, high accuracy model for traffic sign recognition
* architecture from scratch to see how far small-scale, relatively vanilla deep learning methods can be pushed
* simple and portable, not for use in performance critical contexts but perhaps in resource critical ones
* able to be fine-tuned on any number of traffic sign datasets easily without extensive retraining

### How is it done today, and what are the limits of current practice? 

* avoid contamination, didn’t look too much at extant methods
* however many sota solutions borrow from other disciplines 
* transformers, self-attention, e.g. one uses spatial transformer and modified inception module for 99.8% accuracy
* 10.5 million parameters
* others use very large networks
* 90 million parameters for committee of cnn, like random forest but for cnn
* obviously high accuracy is possible, want instead to achieve similar accuracy with simple results

### Who cares? If you are successful, what difference will it make? 

* nothing shockingly groundbreaking
* lightweight model has potential for mobile uses, e.g. with smart glasses 
* could load it with country signs before trip and have hud with them on it
* automakers could be interested too 
* not for full self driving but for safety features, e.g. calling out signs to driver just in case, monitoring speed, etc. 

### What data did you use? 

* used well-known gtsrb dataset, publicly available online and used by many, cc0 license
* originally from intl joint converence on neural networks IJCNN 
* 43 classes of german traffic sign, 50,000 images
* cropped to show only 1 example of class but otherwise unprocessed
* labeled with class from 1 to 43 inclusive
* anywhere between 2000 and 150 examples for any given class
* no identifying or sensitive data, completely self-contained, basically it’s a good benchmark

# Approach

### What did you do exactly? How did you solve the problem? Why did you think it would be successful? 

* first conducted data exploration, led to problems + their solutions, used pandas + scikit libs for visualization
* then write code to wrangle dataset, preprocess, split into train/val/test
* tons of data so val is useful as a sort of “intermediate test” to measure potential overfitting without data peeking
* then move to training experimentation
* start small with simple cnn models
* few iterations of conv -> conv -> max pool -> dropout
* not great performance, slowly scale up layers + learned filters
* eventually get to a comfortable place, plug it into a gpu and train for 35 epochs
* architecture diagram below with shapes
* (conv conv maxpool dropout) x 3 -> fc x2, all relu activation except final layer softmax
* include dropout as another way to combat overfitting and increase model resilience
* trained using tensorflow 2.7.0, accelerated on rtx 3060ti, cuda 11.4, cudnn 8.2.1.32
* used categorical crossentropy for loss with onehot encoded ground truth
* batch size of 32 due to memory constraints
* adam optimizer, learning rate of 0.001
* lr reduction callback — at any point if val loss does not decrease for 4 consecutive epochs reduce the lr by 0.1
* allows for better fine tuning as we get closer to minima

### What problems did you anticipate? What problems did you encounter? 

* first problem encountered was class imbalance

* simple graph of class vs # of examples shows big difference in numbers of classes 

* <insert graph here\>

* best represented class has approximately 10x more examples than least represented 

* combat this we weight samples during training 

* divide max # of examples by # examples for class, that’s weight for class 

* e.g. if the max is 2x a class then that class is 2x more likely to be seen

* additionally use data augmentation during training

* random rotations, zoom, x- and y-axis shifts, shears

* also designed to make model more generalizable and smaller classes more represented

* second problem had to do with images themselves

* cropped well but otherwise not preprocessed

* <insert demonstration of bad lighting\>

* you can see lighting for images is what we call in business Very Bad

* some dark, some super overexposed, just bad all around

* use normalization technique clahe to adjust images so they are much more uniform

* histogram equalization at large common preprocessing in computer vision

	

* simple operation, very efficient, low performance impact for something like live video so we leave it alone

* convert image to hsv, split into tiles

* generate histogram for the v of each tile

* clip off any v that’s above a certain threshold

* instead of throwing, redistribute evenly to bins that will not overflow

* calculate cdf for histogram values, scale and map to image using original pixels

* then combine tiles using some kind of interpolation, we use bilinear because like it doesn’t matter lmao

* after clahe you can see images from above are much better 

* not particularly nice to look at but computers like them a lot more

# Experiments and Results

### How did you measure success? What measurements were used? What were results? Succeed/fail/why

* measure success via accuracy metric (# of correct predictions) combined with number of trainable parameters
* want to keep parameters low while keeping accuracy high, kind of dual optimization problem
* ultimately achieved 96.54% accuracy on test dataset with only 103,627 parameters
* other networks used 100x or 1000x that for like 3% more accuracy
* consider this a success

* [confusion matrix discussion and example of misclassified images]
