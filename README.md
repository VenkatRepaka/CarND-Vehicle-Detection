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

### Feature selection
I have chosen features of YCrCb color space features and spatial binning of (32, 32)
Below is a image showing the features
![color_space_image_spatial](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/color_space_image_spatial.jpg)


### Hog Parameters
Below are the final HOG parameters that I have chosen which gave the best accuracy.

| Parameter      | Value     |
|----------------|-----------|
| Color Space    | YCrCb     |
| Spatial Size   | 32 X 32   |
| Histogram Bins | 32/64/128 |
| Hog Pixels     | 8         |
| Hog Cells      | 2         |
| Hog Channel    | ALL       |

Below is a sample of hog features for a car and non car
![hog_features](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/hog_features.jpg)

### Classifier
I have used LinearSVC as classifier for the project. The training process is part of [classifier.py](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/vd/classifier.py)
The classifier data is stored in model_params.p using pickle.

Below are the predictions of classifier with false positives on test images with ystart 400 and ystop 600 and scales 1, 1.3, 1.5, 1.8, 2, 2.4

![test1](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/prediction_test1.jpg)
![test2](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/prediction_test2.jpg)
![test3](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/prediction_test3.jpg)
![test4](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/prediction_test4.jpg)
![test5](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/prediction_test5.jpg)
![test6](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/prediction_test6.jpg)


### Sliding windows
Below are the example of sliding windows applied using scales 1, 1.3, 1.5, 1.8, 2, 2.4
![scale1](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/scale_1.jpg)
![scale1.3](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/scale_1.3.jpg)
![scale1.5](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/scale_1.5.jpg)
![scale1.8](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/scale_1.8.jpg)
![scale2](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/scale_2.jpg)
![scale2.4](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/scale_2.4.jpg)


### Image pipeline
below are the images after removing false positives and applying heatmap

![heatmap1](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/heatmap_test1.jpg)
![heatmap2](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/heatmap_test2.png)
![heatmap3](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/heatmap_test3.png)
![heatmap4](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/heatmap_test4.png)
![heatmap5](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/heatmap_test5.png)
![heatmap6](https://github.com/VenkatRepaka/CarND-Vehicle-Detection/blob/master/documentation/heatmap_test6.png)


#### Discussion
Many of the bounding boxes are not completely enclosing the cars in the video. Changing the thresholding will help but this is increasing the false positives.

The speed of the process involved here is extremely slow.

