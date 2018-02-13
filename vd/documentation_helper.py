import glob
from vd.lectures_functions import *
import random
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt


cars = glob.glob('../vehicles/**/*.png')
non_cars = glob.glob('../non-vehicles/**/*.png')
cars_index = random.randint(0, len(cars))
non_cars_index = random.randint(0, len(non_cars))

car_image = mpimg.imread(cars[cars_index])
car_image_YCrCb = cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)

non_car_image = mpimg.imread(non_cars[non_cars_index])
non_car_image_YCrCb = cv2.cvtColor(non_car_image, cv2.COLOR_RGB2YCrCb)

car_hist_features = color_hist(car_image_YCrCb)
non_car_hist_features = color_hist(non_car_image_YCrCb)

car_bin_spatial = bin_spatial(car_image_YCrCb)
non_car_bin_spatial = bin_spatial(non_car_image_YCrCb)

f, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(12, 6))
f.tight_layout()
ax1.imshow(car_image)
ax1.set_title('Car Image', fontsize=10)
ax2.imshow(car_image_YCrCb)
ax2.set_title('YCrCb Car Image', fontsize=10)
ax3.plot(car_hist_features)
ax3.set_title('Car Histogram', fontsize=10)
ax4.imshow(cv2.resize(car_bin_spatial, (32, 32)))
ax4.set_title('YCrCb Car Spatial (32, 32)', fontsize=10)
ax5.plot(car_bin_spatial)
ax5.set_title('YCrCb Car Spatial (32, 32) features', fontsize=10)

ax6.imshow(non_car_image)
ax6.set_title('Non Car Image', fontsize=10)
ax7.imshow(non_car_image_YCrCb)
ax7.set_title('YCrCb Non Car Image', fontsize=10)
ax8.plot(non_car_hist_features)
ax8.set_title('Non Car Histogram', fontsize=10)
ax9.imshow(cv2.resize(non_car_bin_spatial, (32, 32)))
ax9.set_title('YCrCb Non Car Spatial (32, 32)', fontsize=10)
ax10.plot(non_car_bin_spatial)
ax10.set_title('YCrCb Non Car Spatial (32, 32) features', fontsize=10)
plt.show()

orient = 9
pix_per_cell = 8
cell_per_block = 2
car_hog_features, car_hog_image = get_hog_features(cv2.cvtColor(car_image, cv2.COLOR_RGB2GRAY), orient, pix_per_cell,
                                                   cell_per_block, vis=True, feature_vec=True)
non_car_hog_features, non_car_hog_image = get_hog_features(cv2.cvtColor(non_car_image, cv2.COLOR_RGB2GRAY), orient,
                                                           pix_per_cell, cell_per_block, vis=True, feature_vec=True)
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 6))
ax1.imshow(car_image)
ax1.set_title('Car Image', fontsize=10)
ax2.imshow(car_hog_image, cmap='gray')
ax2.set_title('Car Hog', fontsize=10)
ax3.plot(car_hog_features)
ax3.set_title('Car Hog Histogram', fontsize=10)
ax4.imshow(non_car_image)
ax4.set_title('Non Car Image', fontsize=10)
ax5.imshow(non_car_hog_image, cmap='gray')
ax5.set_title('Non Car Hog', fontsize=10)
ax6.plot(non_car_hog_features)
ax6.set_title('Non Car Hog Histogram', fontsize=10)
plt.show()


