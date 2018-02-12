from exprtimentation.lesson_functions import *
from scipy.ndimage.measurements import label
import numpy as np
from moviepy.editor import VideoFileClip
import pickle


classifier_pickle = pickle.load(open('classifier.p', 'rb'))
svc = classifier_pickle["scv"]
X_scaler = classifier_pickle["scaler"]
orient = classifier_pickle["orient"]
pix_per_cell = classifier_pickle["pix_per_cell"]
cell_per_block = classifier_pickle["cell_per_block"]
spatial_size = classifier_pickle["spatial_size"]
hist_bins = classifier_pickle["hist_bins"]


def pipeline(image):
    # global heat_p
    hot_windows = []
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    hot_windows += find_cars(image, 350, 650, 1.0, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, conv='RGB2YUV')
    hot_windows += find_cars(image, 350, 650, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, conv='RGB2YUV')
    hot_windows += find_cars(image, 350, 650, 2.0, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, conv='RGB2YUV')
    hot_windows += find_cars(image, 350, 650, 3.0, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, conv='RGB2YUV')
    heatmap_img = np.zeros_like(image[:, :, 0])
    heatmap_img = add_heat(heatmap_img, hot_windows)
    heatmap_img = apply_threshold(heatmap_img, 3)
    labels = label(heatmap_img)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img


video_output = '../test_video_output.mp4'
clip = VideoFileClip("../test_video.mp4")

project_clip = clip.fl_image(pipeline)
project_clip.write_videofile(video_output, audio=False)
