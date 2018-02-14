from vd.lectures_functions import *
import glob
import matplotlib.pyplot as plt
import pickle


# global svc, X_scaler,orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, ystart, ystop
test_images = glob.glob('../test_images/*.jpg')
dist_pickle = pickle.load(open("./model_params.p", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
ystart = 400
ystop = 600

for img in test_images:
    image = mpimg.imread(img)
    scales = [1, 1.3, 1.5, 1.8, 2, 2.4]
    hot_windows = None
    for scale in scales:
        if hot_windows is None:
            hot_windows = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                    spatial_size, hist_bins)
        else:
            hot_windows = hot_windows + find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                                  cell_per_block, spatial_size, hist_bins)

    bbox_list, heatmap = process_bboxes(image, hot_windows, threshold=1, show_heatmap=True)
    draw_img = draw_car_boxes(image, bbox_list)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(draw_img)
    ax1.set_title('Detections', fontsize=50)
    ax2.imshow(heatmap, cmap='hot')
    ax2.set_title('Heatmap', fontsize=50)
    plt.show()
