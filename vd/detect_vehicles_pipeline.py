from vd.lectures_functions import *
import pickle
from moviepy.editor import VideoFileClip
import collections


dist_pickle = pickle.load( open("./model_params.p", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
y_start_stop = [400, 600]  # Min and max in y to search in slide_window()
ystart = y_start_stop[0]
ystop = y_start_stop[1]

queued_boxes = collections.deque(maxlen=10)


def process_image(image):
    scales = [1, 1.3, 1.5, 1.8, 2, 2.4, 3]
    # Apply all scales
    hot_windows = None
    for scale in scales:
        if hot_windows is None:
            hot_windows = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                    spatial_size, hist_bins)
        else:
            hot_windows = hot_windows + find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                                  cell_per_block, spatial_size, hist_bins)
    # Identify boxes using threshold to avoid false positives
    bbox_list = process_bboxes(image, hot_windows, threshold=1, show_heatmap=False)
    # Add to buffered list
    queued_boxes.append(bbox_list)
    # Get all queued boxes from last n frames
    avg_bbox_list = []
    for box in queued_boxes:
        for b in box:
            avg_bbox_list.append(b)
    # Averaging - Apply avg heatmap to reduce false positives and smooth the bounding boxes
    bbox_list, heatmap = process_bboxes(image, avg_bbox_list, threshold=7, show_heatmap=True)
    draw_img = draw_car_boxes(image, bbox_list)
    return draw_img


vid_output = '../project_video_result.mp4'
project_video = VideoFileClip('../project_video.mp4')
result_video = project_video.fl_image(process_image)
result_video.write_videofile(vid_output, audio=False)
