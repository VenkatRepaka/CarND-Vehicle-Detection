from vd.lectures_functions import *
import pickle
from moviepy.editor import VideoFileClip
from vd.vehicle_detector import VehicleDetector


dist_pickle = pickle.load( open("../poan/model_params.p", "rb"))
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

vehicle_detector = VehicleDetector(10)


def process_image(image):
    # if process_image.global_cache is None:
    #     buffered_bboxes = collections.deque(maxlen=n_frames)
    #     actual_boxes = collections.deque(maxlen=n_frames)
    #     global_cache = {
    #         'buffered_bboxes': buffered_bboxes,
    #         'actual_boxes': actual_boxes
    #     }
    # else:
    #     global_cache = process_image.global_cache
    #     buffered_bboxes = global_cache['buffered_bboxes']
    #     actual_boxes = global_cache['actual_boxes']
    # The scales of sliding windows that HOG features will be computed
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
    bbox_list = process_bboxes(image, hot_windows, threshold=0, show_heatmap=False)
    # Add to buffered list
    vehicle_detector.queued_boxes.append(bbox_list)
    # Append to global cache
    # global_cache['buffered_bboxes'] = buffered_bboxes
    # process_image.global_cache = global_cache
    # Get all buffered boxes
    avg_bbox_list = []
    for box in vehicle_detector.queued_boxes:
        for b in box:
            avg_bbox_list.append(b)
    # Smoothing - Apply avg heatmap to further reduce false positives and smooth the bounding boxes
    bbox_list, heatmap = process_bboxes(image, avg_bbox_list, threshold=7, show_heatmap=True)
    draw_img = draw_car_boxes(image, bbox_list)

    # # Merge heatmap with image
    # sizeX = int(256 * 1.3)
    # sizeY = int(144 * 1.3)
    # heat2 = cv2.resize(heatmap, (sizeX, sizeY))
    # res_img = cv2.resize(image, (sizeX, sizeY))
    # res_img_gray = cv2.cvtColor(res_img, cv2.COLOR_RGB2GRAY)

    #
    # heat3 = (heat2 / np.max(heat2) * 255).astype(int)
    #
    # res_img_gray_R = res_img_gray  # np.zeros_like(res_img_gray)
    # res_img_gray_R[(heat2 > 0)] = 255
    # # img_mag_thr[(imgThres_yellow==1) | (imgThres_white==1) | (imgThr_sobelx==1)] =1
    # res_img_gray_G = res_img_gray
    # res_img_gray_G[(heat2 > 0)] = 0
    # res_img_gray_B = res_img_gray
    # res_img_gray_B[(heat2 > 0)] = 0
    #
    # draw_img[0:sizeY, 0:sizeX, 0] = res_img_gray_R + heat3
    # draw_img[0:sizeY, 0:sizeX, 1] = res_img_gray
    # draw_img[0:sizeY, 0:sizeX, 2] = res_img_gray

    return draw_img


vid_output = '../poan_mine/project_video_result.mp4'
project_video = VideoFileClip('../project_video.mp4')
result_video = project_video.fl_image(process_image)
result_video.write_videofile(vid_output, audio=False)
