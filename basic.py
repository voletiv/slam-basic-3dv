# sudo pip3 install opencv-python==3.4.2.16
# sudo pip3 install opencv-contrib-python==3.4.2.16

import cv2
import numpy as np

video_file = 'vids/duckietown.mp4'

cap = cv2.VideoCapture(video_file)

# SIFT
sift = cv2.xfeatures2d.SIFT_create()

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

while(1):
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(old_frame, None)
    kp2, des2 = sift.detectAndCompute(frame_gray, None)
    # # Draw keypoints
    # img1 = cv2.drawKeypoints(old_frame, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # img2 = cv2.drawKeypoints(frame_gray, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # FLANN matches
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    # draw the tracks
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    cv2.putText(mask, str(len(good)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)
    k = cv2.waitKey(300) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()

cap.release()


p1 = np.array([[1, 1, 1], [2, 2, 1], [3, 3, 1], [4, 4, 1], [5, 5, 1], [6, 6, 1], [7, 7, 1], [8, 8, 1]], dtype=float)
p1[:, :2] = (p1[:, :2] - 5.)/5.

np.

p2 = p1 + [.1, 0, 0]

K = np.array([[10, 0, -5], [0, 10, -5], [0, 0, 1]], dtype=float)

KinvP1 = np.matmul(np.linalg.inv(K), p1.T).T
KinvP2 = np.matmul(np.linalg.inv(K), p2.T).T
