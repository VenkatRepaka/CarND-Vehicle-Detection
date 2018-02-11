import matplotlib
matplotlib.use('TkAgg')
import pickle
from vehicle_detection.lesson_functions import *
import glob
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label


classifier_pickle = pickle.load(open('classifier.p', 'rb'))
svc = classifier_pickle["scv"]
X_scaler = classifier_pickle["scaler"]
orient = classifier_pickle["orient"]
pix_per_cell = classifier_pickle["pix_per_cell"]
cell_per_block = classifier_pickle["cell_per_block"]
spatial_size = classifier_pickle["spatial_size"]
hist_bins = classifier_pickle["hist_bins"]

ystart = 400
ystop = 650
scale = 0.5

images = glob.glob('../test_images/test*.jpg')
for img in images:
    image = mpimg.imread(img)
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    hot_windows = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins)
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    plt.show()