import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in an image
# You can also read cutout2, 3, 4 etc. to see other examples
image = mpimg.imread('../class_notes_test_images/test_img.jpg')


# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
# KEEP IN MIND IF YOU DECIDE TO USE THIS FUNCTION LATER
# IN YOUR PROJECT THAT IF YOU READ THE IMAGE WITH
# cv2.imread() INSTEAD YOU START WITH BGR COLOR!
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features


feature_vec_rgb = bin_spatial(image, color_space='RGB', size=(32, 32))
feature_vec_hsv = bin_spatial(image, color_space='HSV', size=(32, 32))
feature_vec_luv = bin_spatial(image, color_space='LUV', size=(32, 32))
feature_vec_hls = bin_spatial(image, color_space='HLS', size=(32, 32))
feature_vec_yuv = bin_spatial(image, color_space='YUV', size=(32, 32))
feature_vec_ycrcb = bin_spatial(image, color_space='YCrCb', size=(32, 32))

# Plot features
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 3))
fig.tight_layout()
ax1.plot(feature_vec_rgb)
ax1.set_title('RGB')
ax2.plot(feature_vec_hsv)
ax2.set_title('HSV')
ax3.plot(feature_vec_luv)
ax3.set_title('LUV')
ax4.plot(feature_vec_hls)
ax4.set_title('HLS')
ax5.plot(feature_vec_yuv)
ax5.set_title('YUV')
ax6.plot(feature_vec_ycrcb)
ax6.set_title('YCrCb')
plt.show()