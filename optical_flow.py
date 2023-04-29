import numpy as np
import cv2 as cv

cap = cv.VideoCapture(
    '../data/audio(wideo)/non_animal/C_non-contrast_1/C_non-contrast_1_processed/REC64.MP4')
optical_flow_params = dict(winSize=(30, 30), maxLevel=2)
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
cv.imshow('image', old_frame)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
start_point = [51., 200.]  # x and y
p0 = np.array([[start_point]]).astype(np.float32)

mask = np.zeros_like(old_frame)
while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **optical_flow_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)),
                       (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
cv.destroyAllWindows()
