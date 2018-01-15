CarND · T1 · P5 · Vehicle Detection Project Writeup
===================================================


[//]: # (Image References)

[image1]: ./output/images/001%20-%20Cars.png "Cars Examples"
[image2]: ./output/images/002%20-%20Non%20Cars.png "Non Cars Examples"
[image3]: ./output/images/003%20-%20Hog%20Car.png "Hog Car Example"
[image4]: ./output/images/004%20-%20Hog%20Non%20Car.png "Hog Non Car Example"
[image5]: ./output/images/008%20-%20All%20Grids.jpg "All Grids"
[image6]: ./output/images/009%20-%20Raw%20Detections.png "Raw Detections"
[image7]: ./output/images/010%20-%20Heatmaps.png "Heatmaps"
[image8]: ./output/images/011%20-%20Thresholded%20Heatmaps.png "Thresholded Heatmaps"
[image9]: ./output/images/012%20-%20Detected%20Labels.png "Detected Labels"
[image10]: ./output/images/013%20-%20Detected%20Boxes.png "Detected Boxes"
[image11]: ./output/images/014%20-%20Final%20Result.jpg "Final Result"


Project Goals
-------------

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


Rubric Points
-------------

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  


### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in [Feature Extraction & Classifier Creation and Training.ipynb](src/notebooks/Feature%20Extraction%20%26%20Classifier%20Creation%20and%20Training.ipynb).

Half way through [Section 1 - Load Data](src/notebooks/Feature%20Extraction%20%26%20Classifier%20Creation%20and%20Training.ipynb#1.-LOAD-DATA), there are some examples of raw car and non car images:

![Cars Examples][image1]

![Non Cars Examples][image2]

Down below, in [Section 2 - Extract Features](src/notebooks/Feature%20Extraction%20%26%20Classifier%20Creation%20and%20Training.ipynb#2.-EXTRACT-FEATURES), I preview some HOG feature examples for a car and a non-car `HLS` image, generated with parameters `orientations=9`, `pixels_per_cell=(12, 12)` and `cells_per_block=(2, 2)`:

![Hog Car Example][image3]

![Hog Non Car Example][image4]


#### 2. Explain how you settled on your final choice of HOG parameters.

As the results of the example classifiers that we saw on the lectures were already quite accurate and I had alreday played around with HOG's params there, I just used those when I created my new model, although I tried various other options, as I will explain in the next point.

I didn't write down the whole set of combinations that I tried, but the one that worked the best with all 3 channels of HSV were:
- `9` orientations.
- `12` pixels per cell.
- `2` cells per block

However, I ended up using a different color space (HLS) and assuming those values would still work good with it, which I actually verified by visually inspecting the results on the test video.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The feature extraction and training of my classifier all happens in [Feature Extraction & Classifier Creation and Training.ipynb](src/notebooks/Feature%20Extraction%20%26%20Classifier%20Creation%20and%20Training.ipynb), especifically in [Section 2 - Extract Features](src/notebooks/Feature%20Extraction%20%26%20Classifier%20Creation%20and%20Training.ipynb#2.-EXTRACT-FEATURES) and [Section 3 - Train Classifier (SVM)](src/notebooks/Feature%20Extraction%20%26%20Classifier%20Creation%20and%20Training.ipynb#3.-TRAIN-CLASSIFIER-(SVM)), respectively.

The parameters used for the feature extraction: spatial binning, histograms of color and HOG features, were all adjusted in the online Udacity playgrounds that we saw during the lectures. The same parameters I found usefull there were used here, even though I ended up using HLS color space, instead of the one I was using initially, HVS, that was the one those params were adjusted for.

The datasets were split in training and test 80%/20%. Then, a `Pipeline` as created in order to hold both the scaler (`StandardScaler`) and the classifier (`LinearSVC`) and be able to save/load both of them together (last cell of the Notebook).

Once the classifier is trained, the Notebook will output its training time, training and test accuracies, prediction time and confusion matrix. Based on those metrics, I tried out different classifiers with different params. Without getting into too much detail about the params here (they can be found in a comment in the corresponding cell in the Notebook), the classifiers that I have tried are:

- `LinearSVC`.
- `SVC` with linear kernel.
- `SVC` with rbf kernel.
- `SVC` with poly kernel.
- `DecisionTreeClassifier`.
- `RandomForestClassifier`.
- `AdaBoostClassifier`.
- `GaussianNB`.

All but `GaussianNB` produced models with accuracies above 95% and the `SVC` with rbf kernel got a 99.13% accuracy in test! However, `SVC` was slow both training and making predictions. Actually, all the classifiers (except `GaussianNB`) were slower than `LinearSVC`, being `AdaBoostClassifier` the slowest one.

I finally decided to use `LinearSVC` with `loss="hinge"` as it was getting a 98.59% accuracy while training and predicting in a relatively short time.

Later, by checking the results produced by this model on video, I tried training it again using different color spaces: RGB, HSV, LUV, HLS, YUV and YCR_CB (YCrCb) and saw HLS was the one producing less false positives, so I replaced HSV with it.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

I created 3 window grids of different sizes:

<table>
  <tr>
    <td><img src="output/images/005 - XS Grid.jpg" alt="XS Grid" /></td>
    <td><img src="output/images/006 - S Grid.jpg" alt="S Grid" /></td>
    <td><img src="output/images/007 - M Grid.jpg" alt="M Grid" /></td>
  </tr>
  <tr>
    <td>XS - `64px` - `50% overlap`</td>
    <td>S - `96px` - `75% overlap`</td>
    <td>MS - `128px` - `75% overlap`</td>
  </tr>
</table>

And this is how all 3 come together:

![All Grids][image5]

This was done by trial and error by inspecting some example frames from the video and finding positions, dimensions and overlap percentages that produced grids that could fit the locations and dimensions of the cars on those example frames, as can be seen in [Sliding Window Setup](src/notebooks/Sliding%20Window%20Setup.ipynb) step by step.

Initially, a forth grid of 256px was created, but that one proved to work really bad on video streams, producing only false positives most of the time, so it was finally removed. A reason for this might be that the non-car examples available to train our classifier do not include images similar to the ones that would be generated by such a big window.

Although that issue could have been addressed with hard negative mining, as it was done with the other false detections, such a big window would not be really helpful in this video (it could have been to detect cars that are driving just in front of ours or in narrower, curvy roads), and would have been processing slower.

Once I was happy with the result, the final configuration was added to [constants.py:15-25](src/helpers/constants.py).


#### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

Here are some example predictions before filtering them with heatmaps, as returned by [`finder.py:10-45 (find_cars)` ](src/helpers/finder.py):

![Raw Detections][image6]

As mentioned before, plenty of different classifiers, params and color spaces have been considered and tested in order to generate a model that is either accurate and fast making predictions. However, the model accuracy on its own was not really helpign to much to distinguish those models that would perform better (less false positives) on videos, so they had to be verified visually on images and videos.

A few actions have been taken in order to reduce the number of false positives.

On one hand, `svc.decision_function` has been used instead of `svc.predict` ([`finder.py:36-43 (find_cars)` ](src/helpers/finder.py)), which returns the distance of the samples X to the separating hyperplane. That distance is then thresholded so that a match is only considered if the distance is greater than 0.6, while the behavior implemented in `svc.predict` would consider a match anything that is above 0.

On the other hand, a round of hard negative mining has been done with one of the latest versions of the model (already refined and adjusted using other methods), on each of the 3 short videos available in this project under `input/videos` (the one provided in the initial project and two more I added with problematic portions of the video). That was done automatically with the `extract_false_positives` function in the [Video Processing Notebook](src/notebooks/Video%20Processing.ipynb), that would consider any detection on the left 60% side of the video a false positive (as there are no cars there), and save them as `png` non-vehicle images that could then be used to retrain the model.

Initially, `428` new images were produced and can be found it [`input\images\dataset\non-vehicles\falses-short`](input\images\dataset\non-vehicles\falses-short). After adding them, there were a total of `8792` (`48.34 %`) car imags and `9396` (`51.66 %`) non-car images. In order to rebalance the dataset, `604` car images need to be added as well, alghouth that difference is not too big. In order to make the model perform better on the project's video, where it had a harder time detecting the white car than it had to detect the black one, `434` white/light car images had been selected from the existing ones and will be added to the car images set after flipping them horitzontally.

However, that was not enough to stop getting false positives on the left side of the road on the full video, especially in the beginning of the videos, that doesn't appear in the short ones, where there are areas with grass and flowers, that is, bright colors. Therefore, another round of hard negative mining was done with the whole full video, generating `XXX` images that were added to the training set.


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](output/videos/004%20-%20project%20Video%20(Full).mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The video processing pipeline has been implemented in [Video Processing.ipynb](src/notebooks/Video%20Processing.ipynb), particularly in the `process_frame` function under [Section 2 - Create The Video Processing Pipeline](src/notebooks/Video%20Processing.ipynb#2.-CREATE-THE-VIDEO-PROCESSING-PIPELINE). Its steps are:

- Record the positions of positive detections in each frame of the video.
- Those new detections are appended to a `deque` inside the class `VehicleTracker`, also implemented in that same cell.
- From all the positive detections inside that `deque`, it creates a heatmap and then thresholds that map to identify vehicle positions.
- Next, the function `find_boxes` is called passing it those heatmaps. It will use `scipy.ndimage.measurements.label()` to identify individual blobs in it.
- I then assumed each blob corresponded to a vehicle and constructed bounding boxes to cover the area of each of them.

Here we can see the individual heatmaps from all the 8 images wes saw before, before applying a threshold:

![Heatmaps][image7]

After applying a threshold of 2 to each of them, they look like this:

![Thresholded Heatmaps][image8]

This is the output of  `scipy.ndimage.measurements.label()` on each of them:

![Detected Labels][image9]

And the resulting bounding boxes on each of them:

![Detected Boxes][image10]

In the video, those heatmaps are generated using the detections from the N last frames and the threshold is adjusted based on the amount of frames stored. That produces better detection even when a car is missed in one or even a few subsequent frames, and also helps removing false positives by tolerating higher threshold values.

The final result on a video will look like this:

![Final Result][image11]


### Discussion

##### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As mentioned multiple times, even after adopting plenty of different possible solutions to avoid false positives, such as hard negative mining and subtle training set augmentation in general, thresholding the predictions, thresholding and averaging the heatmaps... A lot of false detections are produced. That also means that it is hard to find values for the thresholds that find a good balance between filtering out false positives and not missing cars.

Also, all those params are very specific to this particular video. What would happen if in a different one we are doing 50 km/h and a car overtakes us doing 200 km/h? Wouldn't the pipeline threshold be too restrictive to catch it? Probably, and we might argue that it doesn't really matter because that car will be gone in a couple of seconds and because that scenario might be unlikely.

Anyway, it is true that the dataset used to generate the model is not too big, so this might not generalise properly for other videos, as we can imagine after seeing the difficulties to filter out false positives and properly detect the white car in this project. That could be solve by either using a bigger dataset or artificially augmenting the current one (or a combination of both), at the cost of training speed.

Also, more features might be needed in order to improve the real accuracy of the model, which would also make both training and predicting slower. Maybe we could even use more than one color space to generate our feature vectors or create an [ensable](http://scikit-learn.org/stable/modules/ensemble.html) using multiple submodels.

Talking about speed, there are a few things we could do to improve either training speed and prediction speed.

Both could be speeded up by using a different hog function. For example, instead of using `skimage.feature.hog`, we could use [`cv2.HOGDescriptor`](https://docs.opencv.org/3.2.0/d5/d33/structcv_1_1HOGDescriptor.html).

To improve just the former, we could look for different libraries that are faster, but that's already one of the reasons I use `LinearSVC`, as it is the fastest classifier of the ones I tried. We might find some other that can be run on GPU though.

To improve the later, we can try to [reduce the dimensionality of our feature vectors](http://scikit-learn.org/stable/modules/unsupervised_reduction.html), which we could even [do automatically using GridSearchCV](http://scikit-learn.org/stable/auto_examples/plot_compare_reduction.html).

Also, we could implement hog subsampling, as explained in the lectures, as HOG is expensive to compute, and right now we are doing it for every window. Moreover, the number of windows could be reduced by reducing the overlaping percentage, as right now we usually have multiple detections for a single car as the windows are quite close from one another.

We could also try to improve vehicle tracking by actually keeping track of their location and average change in position, so that we could even predict their position if, for example, one car is hidden for a while by another one that is overtaking it. That would allow us to improve filtering by adjusting the hotmap threshold dinamically based on the distance of the detections candidates to one of the existing bounding boxes.

However, I think neural networks are more robust and performant than these techniques, so it might be better to take that approach. For example, [YOLO](https://pjreddie.com/darknet/yolo/) can perform real-time object detection on a way wider variety of objects.

