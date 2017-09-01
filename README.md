# Vehicle Detection

--------------------------------------------------------------------------------

## Table of Contents
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Vehicle Detection](#vehicle-detection)
	- [Table of Contents](#table-of-contents)
	- [Project Structure](#project-structure)
	- [Histogram of Oriented Gradients (HOG)](#histogram-of-oriented-gradients-hog)
		- [1\. HOG feature extraction](#1-hog-feature-extraction)
		- [2\. Choice of HOG parameters.](#2-choice-of-hog-parameters)
		- [3\. classifier training](#3-classifier-training)
	- [Sliding Window Search](#sliding-window-search)
		- [1\. Slide window implementation](#1-slide-window-implementation)
		- [2\. Vehicle detection pipeline on single image](#2-vehicle-detection-pipeline-on-single-image)
	- [Video Implementation](#video-implementation)
		- [1\. Vehicle detection on video](#1-vehicle-detection-on-video)
		- [2\. False positive filtering and smoothing](#2-false-positive-filtering-and-smoothing)
	- [Discussion](#discussion)

<!-- /TOC -->

--------------------------------------------------------------------------------

## Project Structure

My project includes the following files:

- vehicle_tracking.py: containing the core vehicle detection functions and script to run the vehicle detection pipeline on a video
- pipeline.ipynb: demonstrate the intermediate results of the pipeline
- video_output.mp4 is the video output of running `vehicle_tracking.py` on `project_video.mp4`
- classifier_model.p contains the trained vehicle classifier model. It'll be loaded by `vehicle_tracking.py`
- README.md summarizing the results

## Histogram of Oriented Gradients (HOG)

### 1\. HOG feature extraction

The code for this step is contained in the in function `extract_feature` around lines #60 through #88 of the file called `vehicle_tracking.py`). The original input image was in RGB form with channel values between 0 and 1\. The image was turned into both `HSV` space and gray scale. HOG feature was then extracted **on the 4 channels including HSVG** with the parameters `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

Here is the result of HOG feature extraction performed on example images of `vehicle` and `non-vehicle` classes:

![alt text][image1]

### 2\. Choice of HOG parameters.

I tried various combinations of color space, number of HOG channels and HOG orientations on a linear SVM classifier. I found the general trend of larger number of features lead to higher classification accuracy. Using HOG feature on one channel give about 94% accuracy, while HOG on 3 channels raise the accuracy to about 99%. Changing number of orientations in HOG from 9 to 12 increase the accuracy to about 99.3%. Choice of color space affect both the HOG feature and color histogram feature, introducing about 1% variation to the accuracy.

- HSV: 99.40%
- RGB: 98.16%
- HLS: 99.20%
- YCrCb: 98.99%

In the end, I decided to use the following parameters for the feature extraction:

- Color Space: HSV
- HOG channels: HSVG (HSV + gray scale)
- HOG orientations: 12
- Combination of HOG features, color histograms, and spatial feature.

### 3\. classifier training

I trained a Logistic Regression classifier in function `vehicle_classifier` in line #122 through #164 in `vehicle_tracking.py`. I first split the dataset to training and test set with 0.8/0.2 ratio. For each training/test image, a 10272 dimensional feature vector was extracted by the aforementioned parameters. I tried linear SVM, SVM with 2nd order polynomial kernel, and Logistic regression, and got the following accuracy on the test set.

- linear SVM: 99.27%
- LogisticRegression: 99.40%
- SVM: poly =2: 99.62%

SVM with 2nd order polynomial kernel achieved the best accuracy, however the time it takes to make prediction is 50ms, which is too slow. Linear SVM and Logistic both take about 0.1ms to make prediction, and the accuracy were comparable. I decided to use **logistic regression** in the end, because it is able to predict probability. For each type of classifier, I varied the normalization factor (C) from 0.1 to 10, and find no major impact.

## Sliding Window Search

### 1\. Slide window implementation

I employed the slide window search with HOG subsampling similar to the one in lecture. The code is in function `car_search_single_scale` in `vehicle_tracking.py`. In the function, I used 2 cells_per_step, corresponding to 75% overlap. By observing the size of vehicles in the project video, I decided to search a list of scales `[0.75, 1.0, 1.25, 1.5, 2.0]`.

### 2\. Vehicle detection pipeline on single image

During the multi scale slide window search, I'm predicting the **probability** of a patch being vehicle, rather than a binary value. I then obtain the bonding box of the vehicles with the help of heat map and probability map. The heat map stores how many times a region is identified as positive (vehicle) and the probability map stores the maximum probability that this region was identified as positive during the all the slide window searches under different scales. Another advantage of probabilistic prediction is I can specify a probability thresholded in heat map accumulation. After trying values ranging from 0.2 to 0.9, I found a p threshold of 0.7 give the optimal result. The heat map and probability map were generated by function `generate_heat_map`.

Finally, I applied threshold on the heat map, and used `scipy.ndimage.measurements.label()` function to separate the connected regions. For each region, I check the maximum probability stored in probability map, and only consider the region vehicle if the value is larger than 0.9\. In other words, in order for a region to be identified as vehicle, one of its sub-region need to have more than 90% probability of being vehicle.

Here's the vehicle detection pipeline running on the stand along test images.

![alt text][image2]

## Video Implementation

### 1\. Vehicle detection on video

Here's the result of running the pipeline on the project video.

<img src="./output_images/video_output.gif" width="720"/>

Here's a [link to my video result][video1]

### 2\. False positive filtering and smoothing

The steps for false positive filtering was discussed in the single image pipeline. When applying the pipeline on the project video, I performed smoothing on the heat map and probability map between the frames. I first extract the maps `m[0]` on the current frame `f[0]`, then set the current map to the wighted average of current map and map of previous step:

```python
decay_factor = 0.3
m[0] = decay_factor * m[0] + (1-decay_factor) * m[-1]
```


--------------------------------------------------------------------------------

## Discussion

One issue of my current project is I should have been more careful in training/test data split the classifier training step. As mentioned in the project tips, some images were nearly identical in the data set, leading to the contamination of test set. As a result, the classifier model is likely to be overfitting. That's probably part of the reason why I see more features lead to higher accuracy.

Another improvement to the current project might be adding optical flow to help tracking the vehicles and make the bonding boxes more stable.   

[//]: # "Image References"
[image1]: ./output_images/fig1_HOG.png
[image2]: ./output_images/fig2_single_image.png
[video1]: ./output_images/video_output.mp4
