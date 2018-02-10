import pickle
import glob
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from vehicle_detection.lesson_functions import *


color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

classifier_pickle = pickle.load(open("./classifier.p", "rb"))
svc = classifier_pickle["scv"]
X_scaler = classifier_pickle["scaler"]
orient = classifier_pickle["orient"]
pix_per_cell = classifier_pickle["pix_per_cell"]
cell_per_block = classifier_pickle["cell_per_block"]
spatial_size = classifier_pickle["spatial_size"]
hist_bins = classifier_pickle["hist_bins"]

# Test the classifier
start_time = time.time()
test_images = glob.glob('../test_images/test1.jpg')
drawn_images = []
for test_image in test_images:
    image = mpimg.imread(test_image)
    draw_image = np.copy(image)
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[350, 640],
                           xy_window=(256, 128), xy_overlap=(0.85, 0.85))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)
    print('No of windows ', len(windows), ' No of hot windows ', len(hot_windows))
    # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    # drawn_images.append(window_img)

for image in drawn_images:
    plt.imshow(image)
    plt.show()
