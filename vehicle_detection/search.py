import pickle
from vehicle_detection.lesson_functions import *
import numpy as np
import matplotlib.pyplot as plt


classifier_pickle = pickle.load(open('classifier.p', 'rb'))
svc = classifier_pickle["scv"]
X_scaler = classifier_pickle["scaler"]
orient = classifier_pickle["orient"]
pix_per_cell = classifier_pickle["pix_per_cell"]
cell_per_block = classifier_pickle["cell_per_block"]
spatial_size = classifier_pickle["spatial_size"]
hist_bins = classifier_pickle["hist_bins"]

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [400, 640]  # Min and max in y to search in slide_window()

image = mpimg.imread('../test_images/test6.jpg')
draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
image = image.astype(np.float32)/255

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                       xy_window=(128, 128), xy_overlap=(0.85, 0.85))

hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                             spatial_size=spatial_size, hist_bins=hist_bins,
                             orient=orient, pix_per_cell=pix_per_cell,
                             cell_per_block=cell_per_block,
                             hog_channel=hog_channel, spatial_feat=spatial_feat,
                             hist_feat=hist_feat, hog_feat=hog_feat)

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

plt.imshow(window_img)
plt.show()
