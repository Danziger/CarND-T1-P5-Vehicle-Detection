CarND · T1 · P5 · Vehicle Detection Project Writeup
===================================================


[//]: # (Image References)

[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

[image0]: ./output/images/001%20-%20Example%20Output.png "Example Output"


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

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]


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

### Sliding Window Search


#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

I created 3 window grids of different sizes:

<table>
  <tr>
    <td><img src="" alt="" /></td>
    <td><img src="" alt="" /></td>
    <td><img src="" alt="" /></td>
  </tr>
  <tr>
    <td>XS - `64px` - `50% overlap`</td>
    <td>S - `96px` - `75% overlap`</td>
    <td>MS - `128px` - `75% overlap`</td>
  </tr>
</table>

And this is how all 3 come together:

![alt text][image3]

This was done by trial and error by inspecting some example frames from the video and finding positions, dimensions and overlap percentages that produced grids that could fit the locations and dimensions of the cars on those example frames, as can be seen in [Sliding Window Setup](src/notebooks/Sliding%20Window%20Setup.ipynb) step by step.

Initially, a forth grid of 256px was created, but that one proved to work really bad on video streams, producing only false positives most of the time, so it was finally removed. A reason for this might be that the non-car examples available to train our classifier do not include images similar to the ones that would be generated by such a big window.

Although that issue could have been addressed with hard negative mining, as it was done with the other false detections, such a big window would not be really helpful in this video (it could have been to detect cars that are driving just in front of ours or in narrower, curvy roads), and would have been processing slower.

Once I was happy with the result, the final configuration was added to [constants.py:15-25](src/helpers/constants.py).


#### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are six frames and their corresponding heatmaps:

![alt text][image5]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image6]

Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image7]


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

