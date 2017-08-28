import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.ndimage.measurements import label

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Define a function to return HOG features and visualization
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(
                                      cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       visualise=vis, feature_vector=feature_vec, block_norm='L2', transform_sqrt=True)
        return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(
        (channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


def extract_feature(img):
    """
    return feature from one img
    img in RGB format
    """

    orient = 12
    pix_per_cell = 8
    cell_per_block = 2
    img = img.astype(np.float32)  # png file has value between 0 and 1
    feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    hog_features = []
    for channel in range(feature_image.shape[2]):
        hog_features.append(get_hog_features(feature_image[:, :, channel],
                                             orient, pix_per_cell, cell_per_block,
                                             vis=False, feature_vec=True))
    hog_features = np.ravel(hog_features)
    hog_features_ = []
    for channel in range(feature_image.shape[2]):
        hog_features_.append(get_hog_features(feature_image[:, :, channel],
                                              orient, pix_per_cell * 2, cell_per_block,
                                              vis=False, feature_vec=True))
    hog_features_ = np.ravel(hog_features_)

    # # Append the new feature vector to the features list
    hist_features = color_hist(feature_image, nbins=32, bins_range=(0, 1))
    spatial_features = bin_spatial(feature_image, size=(16, 16))
    features = np.concatenate((hist_features, spatial_features, hog_features))
    # Return list of feature vectors
    return features


def extract_feature_batch(img_fnames):
    batch_features = []
    for fname in img_fnames:
        img = mpimg.imread(fname)
        batch_features.append(extract_feature(img))
    return batch_features


def vehicle_classifier():
    try:
        with open('classifier_model.p', 'rb') as f:
            data = pickle.load(f)
        model = data['clf']
        X_scaler = data['scaler']
    except IOError:
        print('cant find pre-tranied classifier')
        print('extracting features')
        t = time.time()
        cars = glob.glob('vehicles/**/*.png')
        notcars = glob.glob('non-vehicles/**/*.png')
        car_features = extract_feature_batch(cars)
        notcar_features = extract_feature_batch(notcars)
        t2 = time.time()
        print('finished extracting feature, time elapsed: {}s'.format(t2 - t))
        print('feature vector length: {}'.format(len(car_features[0])))
        X = np.vstack([car_features, notcar_features]).astype(np.float64)
        X_scaler = StandardScaler().fit(X)
        scaled_X = X_scaler.transform(X)
        y = np.hstack((np.ones(len(car_features)),
                       np.zeros(len(notcar_features))))
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=0)
        print('training logistic regression classifier')
        model = LogisticRegression(C=0.1)
        # model = LinearSVC(C=0.1)
        t = time.time()
        model.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train model...')
        print('training Accuracy of model = ', round(
            model.score(X_train, y_train), 4))
        print('Test Accuracy of model = ', round(
            model.score(X_test, y_test), 4))

        with open('classifier_model.p', 'wb') as f:
            pickle.dump({'clf': model, 'scaler': X_scaler}, f)
    return model, X_scaler


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def generate_heat_map(map_size, bbox_list, prob_list):
    # Iterate through list of bboxes
    heatmap = np.zeros(map_size)
    probmap = np.zeros(map_size)
    for box, p in zip(bbox_list, prob_list):
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        if p > 0.5:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        y = probmap[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        y[y<p]=p
    # Return updated heatmap
    return heatmap, probmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels, probmap):
    # Iterate through all detected cars
    color= (100,100, 250)
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        prob = probmap[labels[0] == car_number].mean()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, 6)
        fill_rec = np.array([[bbox[0][0], bbox[0][1]],
                [bbox[0][0]+150, bbox[0][1]],
                [bbox[0][0]+150, bbox[0][1]-30],
                [bbox[0][0], bbox[0][1]-30]])
        cv2.fillConvexPoly(img, fill_rec, color)

        cv2.putText(img,'car{:2.1f}%'.format(prob*100), (bbox[0][0], bbox[0][1]-5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)
    # Return the image
    return img


def car_detection_single_scale(img, ystart, ystop, scale, clf, X_scaler):

    # draw_img = np.copy(img)
    img = img.astype(np.float32) / 255
    orient = 12
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (16, 16)
    hist_bins = 32

    box_list = []
    prob_list = []
    img_tosearch = img[ystart:ystop, :, :]

    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(
            imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell
            #
            # # Extract the image patch
            subimg = cv2.resize(
                ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(
                subimg, nbins=hist_bins, bins_range=(0, 1))

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((hist_features, spatial_features, hog_features)).reshape(1, -1))
            test_prediction = clf.predict_proba(test_features)

            xbox_left = np.int(xleft * scale)
            ytop_draw = np.int(ytop * scale)
            win_draw = np.int(window * scale)
            box_list.append(((xbox_left, ytop_draw + ystart), (xbox_left +
                                                               win_draw, ytop_draw + win_draw + ystart)))
            prob_list.append(test_prediction[0][1])
    return box_list, prob_list


def car_detection_multi_scale(img, ystart, ystop, scales, clf, X_scaler):
    bbox_list = []
    prob_list = []
    for scale in scales:
        bbox, prob = car_detection_single_scale(
            img, ystart, ystop, scale, clf, X_scaler)
        bbox_list = bbox_list + bbox
        prob_list = prob_list + prob
    img_size = (img.shape[0], img.shape[1])
    heatmap, probmap = generate_heat_map(img_size, bbox_list, prob_list)
    heatmap = apply_threshold(heatmap,5.0)
    imcopy = np.copy(img)
    labels = label(heatmap)
    draw_labeled_bboxes(imcopy, labels, probmap)
    return imcopy
