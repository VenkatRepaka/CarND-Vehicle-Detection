## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


### Loading Data
The code for loading the image data to train the classifier is present in classifier.py.
Sample image of a car and a non car
![car_and_non_car](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/car_and_non_car.png)


### Hog Parameters
Below are the final HOG parameters that I have chosen which gave the best accuracy.

| Parameter      | Value     |
|----------------|-----------|
| Color Space    | YCrCb     |
| Spatial Size   | 32 X 32   |
| Histogram Bins | 32        |
| Hog Pixels     | 8         |
| Hog Cells      | 2         |
| Hog Channel    | ALL       |

Below is a sample of a car and hog features
![car_hog_features](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/car_hog.png)

Below is a sample of a non car and hog features
![car_hog_features](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/non_car_hog.png)

### Classifier
I have used LinearSVC as classifier for the project. The training process is part of [classifier.py](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/vehicle_detection/classifier.py)
The classifier data is stored in classifier.p using pickle.

Below are the predictions of classifier on test images with ystart 400, ystop 650 and scale 1.5
![test1](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/classifier_test1.png)
![test2](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/classifier_test2.png)
![test3](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/classifier_test3.png)
![test4](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/classifier_test4.png)
![test5](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/classifier_test5.png)
![test6](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/classifier_test6.png)


### Image pipeline

The ranges I have chosen for different scales
~~~
ystart = 360
ystop = 560
scale = 1.5

ystart = 400
ystop = 500
scale = 1

ystart = 400
ystop = 600
scale = 1.8

ystart = 350
ystop = 700
scale = 2.5
~~~

![heatmap1](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/heatmap_test1.png)
![heatmap2](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/heatmap_test2.png)
![heatmap3](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/heatmap_test3.png)
![heatmap4](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/heatmap_test4.png)
![heatmap5](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/heatmap_test5.png)
![heatmap6](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/heatmap_test6.png)

