import glob
from vehicle_detection.lesson_functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import time
import os


images = glob.glob('../*vehicles/*/*')
# images = glob.glob('../*vehicles_smallset/*/*')
cars = []
notcars = []
for image in images:
    if 'non' in image:
        notcars.append(image)
    else:
        cars.append(image)
color_spaces = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
orients = [8, 9, 10, 12]
pix_per_cells = [8, 12, 16, 32]
cells_per_blocks = [2, 4]
hog_channels = [0, 1, 2, 'ALL']
spatial_sizes = [(16, 16), (32, 32)]
hist_bins_arr = [8,16,32, 64]
xy_windows = [(100, 100), (100, 85), (128, 128), (95, 85)]
xy_overlaps = [(0.5, 0.5), (0.55, 0.55), (0.6, 0.6), (0.7, 0.7), (0.75, 0.75), (0.8, 0.8), (0.85, 0.85), (0.9, 0.9)]

# color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# orient = 9  # HOG orientations
# pix_per_cell = 8  # HOG pixels per cell
# cell_per_block = 2  # HOG cells per block
# hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
# spatial_size = (16, 16)  # Spatial binning dimensions
# hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [450, 650]

test_images = glob.glob('../test_images/test*.jpg')

for color_space in color_spaces:
    for orient in orients:
        for pix_per_cell in pix_per_cells:
            for cell_per_block in cells_per_blocks:
                for hog_channel in hog_channels:
                    for spatial_size in spatial_sizes:
                        for hist_bins in hist_bins_arr:
                            for xy_window in xy_windows:
                                for xy_overlap in xy_overlaps:
                                    car_features = extract_features(cars, color_space=color_space,
                                                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                                                    orient=orient, pix_per_cell=pix_per_cell,
                                                                    cell_per_block=cell_per_block,
                                                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                                    hist_feat=hist_feat, hog_feat=hog_feat, flip=False)
                                    car_features += extract_features(cars, color_space=color_space,
                                                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                                                     orient=orient, pix_per_cell=pix_per_cell,
                                                                     cell_per_block=cell_per_block,
                                                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                                     hist_feat=hist_feat, hog_feat=hog_feat, flip=True)
                                    notcar_features = extract_features(notcars, color_space=color_space,
                                                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                                                       orient=orient, pix_per_cell=pix_per_cell,
                                                                       cell_per_block=cell_per_block,
                                                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                                       hist_feat=hist_feat, hog_feat=hog_feat, flip=False)
                                    notcar_features += extract_features(notcars, color_space=color_space,
                                                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                                                        orient=orient, pix_per_cell=pix_per_cell,
                                                                        cell_per_block=cell_per_block,
                                                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                                        hist_feat=hist_feat, hog_feat=hog_feat, flip=True)

                                    print('No of cars ', len(car_features), ' No of non cars ', len(notcar_features))
                                    # Create an array stack of feature vectors
                                    X = np.vstack((car_features, notcar_features)).astype(np.float64)

                                    # Define the labels vector
                                    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

                                    # Split up data into randomized training and test sets
                                    rand_state = np.random.randint(0, 100)
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.2, random_state=rand_state)

                                    # Fit a per-column scaler
                                    X_scaler = StandardScaler().fit(X_train)
                                    # Apply the scaler to X
                                    X_train = X_scaler.transform(X_train)
                                    X_test = X_scaler.transform(X_test)

                                    print('Using:', orient, 'orientations', pix_per_cell,
                                          'pixels per cell and', cell_per_block, 'cells per block')
                                    print('Feature vector length:', len(X_train[0]))
                                    # Use a linear SVC
                                    svc = LinearSVC(loss='hinge')
                                    # Check the training time for the SVC
                                    t = time.time()
                                    svc.fit(X_train, y_train)
                                    t2 = time.time()
                                    print(round(t2 - t, 2), 'Seconds to train SVC...')
                                    # Check the score of the SVC
                                    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
                                    # Check the prediction time for a single sample
                                    t = time.time()

                                    directory = './classifier_test/samples/' + color_space + '_' + str(orient) + '_' + str(pix_per_cell) + '_' \
                                                + str(cell_per_block) + '_' + str(hog_channel) + '_' + str(spatial_size[0]) + '_' + str(spatial_size[1]) + '_' + str(hist_bins) + '_' \
                                                + str(xy_window[0]) + '_' + str(xy_window[1]) + '_' + str(xy_overlap[0]) + '_' + str(xy_overlap[1])
                                    if not os.path.exists(directory):
                                        os.makedirs(directory)
                                    for test_image in test_images:
                                        image = mpimg.imread(test_image)
                                        draw_image = np.copy(image)

                                        # Uncomment the following line if you extracted training
                                        # data from .png images (scaled 0 to 1 by mpimg) and the
                                        # image you are searching is a .jpg (scaled 0 to 255)
                                        image = image.astype(np.float32) / 255

                                        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                                               xy_window=xy_window, xy_overlap=xy_overlap)

                                        hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                                                     orient=orient, pix_per_cell=pix_per_cell,
                                                                     cell_per_block=cell_per_block,
                                                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                                     hist_feat=hist_feat, hog_feat=hog_feat)
                                        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
                                        file_path = directory + test_image.replace('../test_images', '')
                                        mpimg.imsave(file_path, window_img)
                                        print('Files saved at ', directory)