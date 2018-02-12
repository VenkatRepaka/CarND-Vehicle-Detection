import pickle
import glob
from scipy.ndimage.measurements import label
from vehicle_detection.feature_extraction import *
# import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


with open('classifier.p', 'rb') as pfile:
    pickle_data = pickle.load(pfile)
    svc = pickle_data['svc']
    X_scaler = pickle_data['X_scaler']
    color_space = pickle_data['color_space']
    orient = pickle_data['orient']
    pix_per_cell = pickle_data['pix_per_cell']
    cell_per_block = pickle_data['cell_per_block']
    spatial_size = pickle_data['spatial_size']
    hist_bins = pickle_data['hist_bins']
    hog_channel = pickle_data['hog_channel']


def pipeline(image):
    ystart = 360
    ystop = 560
    scale = 1.5

    hot_windows = find_cars_boxes(image, ystart, ystop, scale, svc, X_scaler, color_space, orient, pix_per_cell,
                                  cell_per_block, hog_channel, spatial_size, hist_bins)

    ystart = 400
    ystop = 500
    scale = 1

    hot_windows += find_cars_boxes(image, ystart, ystop, scale, svc, X_scaler, color_space, orient, pix_per_cell,
                                   cell_per_block, hog_channel, spatial_size, hist_bins)

    ystart = 400
    ystop = 600
    scale = 1.8
    hot_windows += find_cars_boxes(image, ystart, ystop, scale, svc, X_scaler, color_space, orient, pix_per_cell,
                                   cell_per_block, hog_channel, spatial_size, hist_bins)

    ystart = 350
    ystop = 700
    scale = 2.5
    hot_windows += find_cars_boxes(image, ystart, ystop, scale, svc, X_scaler, color_space, orient, pix_per_cell,
                                   cell_per_block, hog_channel, spatial_size, hist_bins)

    heatmap_img = add_heatmap_and_threshold(image, hot_windows, 3)
    labels = label(heatmap_img)
    draw_image = np.copy(image)
    draw_labeled_bboxes(draw_image, labels)
    return draw_image


video_output = '../project_video_output.mp4'
clip = VideoFileClip("../project_video.mp4")

project_clip = clip.fl_image(pipeline)
project_clip.write_videofile(video_output, audio=False)


# test_images = glob.glob('../test_images/test*.jpg')
# for test_image in test_images:
#     img = mpimg.imread(test_image)
#     draw_image = np.copy(img)
#     ystart = 360
#     ystop = 560
#     scale = 1.5
#
#     hot_windows = find_cars_boxes(img, ystart, ystop, scale, svc, X_scaler, color_space, orient, pix_per_cell,
#                                   cell_per_block, hog_channel, spatial_size, hist_bins)
#
#     ystart = 400
#     ystop = 500
#     scale = 1
#
#     hot_windows = find_cars_boxes(img, ystart, ystop, scale, svc, X_scaler, color_space, orient, pix_per_cell,
#                                   cell_per_block, hog_channel, spatial_size, hist_bins)
#
#     ystart = 400
#     ystop = 600
#     scale = 1.8
#     hot_windows += find_cars_boxes(img, ystart, ystop, scale, svc, X_scaler, color_space, orient, pix_per_cell,
#                                    cell_per_block, hog_channel, spatial_size, hist_bins)
#
#     ystart = 350
#     ystop = 700
#     scale = 2.5
#     hot_windows += find_cars_boxes(img, ystart, ystop, scale, svc, X_scaler, color_space, orient, pix_per_cell,
#                                    cell_per_block, hog_channel, spatial_size, hist_bins)
#
#     heatmap_img = add_heatmap_and_threshold(img, hot_windows, 2)
#     labels = label(heatmap_img)
#     draw_image = np.copy(img)
#     draw_labeled_bboxes(draw_image, labels)
#
#     plt.imshow(draw_image)
#     plt.show()
