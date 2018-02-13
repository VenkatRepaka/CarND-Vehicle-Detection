from vd.lectures_functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import time
import pickle
import glob


cars = glob.glob('../vehicles/**/*.png')
notcars = glob.glob('../non-vehicles/**/*.png')

# Define HOG parameters
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32   # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat,
                                hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat,
                                   hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)

X_scaler = StandardScaler().fit(X)  # Fit a per-column scaler
scaled_X = X_scaler.transform(X)  # Apply the scaler to X

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))  # Define the labels vector

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

init_svc = LinearSVC()
# init_svc.fit(X_train, y_train)
# print('Test Accuracy of SVC = ', round(init_svc.score(X_test, y_test), 4))

# Penalty parameters
penalty_parameters = np.logspace(-6, -1, 10)
# Tolerance parameters
tolerance_parameters = [0.0001, 0.00001]

clf = GridSearchCV(estimator=init_svc, param_grid=dict(C=penalty_parameters, tol=tolerance_parameters))

clf.fit(X_train, y_train)
print(clf.best_score_)
print(clf.best_estimator_)
best_svc = clf.best_estimator_

print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use the best linear SVC of GridSearchSVC
# Check the training time for the SVC
t = time.time()
best_svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(best_svc.score(X_test, y_test), 4))
print(best_svc)

model_params = {
        "svc": best_svc,
        "scaler": X_scaler,
        "orient": orient,
        "pix_per_cell": pix_per_cell,
        "cell_per_block": cell_per_block,
        "spatial_size": spatial_size,
        "hist_bins": hist_bins
       }
pickle.dump(model_params, open('model_params.p', 'wb'))
