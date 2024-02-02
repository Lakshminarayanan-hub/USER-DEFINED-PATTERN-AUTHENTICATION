import cv2, numpy as np, glob
# cv2 - open source computer vision library
# numpy - numerical python library
img1 = cv2.imread('encryptionimg.jpg') # preloading user defined pattern
win_name = 'Camera Matching'
MIN_MATCH = 10
images = glob.glob('*.jpg')
currentImage=0
replaceImg=cv2.imread(images[currentImage])
rows,cols,ch = replaceImg.shape
pts1 = np.float32([[0, 0],[0,rows],[(cols),(rows)],[cols,0]])
zoomLevel = 0
processing = True
maskThreshold=10
detector = cv2.ORB_create(1000)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm = FLANN_INDEX_LSH,table_number = 6,key_size = 12,multi_probe_level = 1)
search_params = dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if img1 is None:
        res = frame
    else:
        img2 = frame
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)
        matches = matcher.knnMatch(desc1, desc2, 2)
        ratio = 0.75
        good_matches = [m[0] for m in matches \
                            if len(m) == 2 and m[0].distance < m[1].distance * ratio]
        matchesMask = np.zeros(len(good_matches)).tolist()
        if len(good_matches) > MIN_MATCH:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask.sum() > MIN_MATCH:
                matchesMask = mask.ravel().tolist()
                h,w, = img1.shape[:2]
                pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
                dst = cv2.perspectiveTransform(pts,mtrx)
                dst = cv2.getPerspectiveTransform(pts1,dst)
                rows, cols, ch = frame.shape
                distance = cv2.warpPerspective(replaceImg,dst,(cols,rows))
                rt, mk = cv2.threshold(cv2.cvtColor(distance, cv2.COLOR_BGR2GRAY), maskThreshold, 1,cv2.THRESH_BINARY_INV)
                mk = cv2.erode(mk, (3, 3))
                mk = cv2.dilate(mk, (3, 3))
                
                for c in range(0, 3):
                    frame[:, :, c] = distance[:,:,c]*(1-mk[:,:]) + frame[:,:,c]*mk[:,:]
        cv2.imshow('img', frame)
        key = cv2.waitKey(1)
        if key == ord('e'):
            print('''Ended
                             Made by Lakshmi Narayanan M''')
            break
        # zoom out
        if key == ord('o'):
            zoomLevel=zoomLevel+0.05
            rows,cols,ch = replaceImg.shape
            A=[-zoomLevel*cols,-zoomLevel*rows]
            B=[-zoomLevel*cols,zoomLevel*rows]
            C=[zoomLevel*cols,zoomLevel*rows]
            D=[zoomLevel*cols,-zoomLevel*rows]
            pts1 = np.float32([[0, 0],[0,rows],[(cols),(rows)],[cols,0]])
            pts1 = pts1 + np.float32([A,B,C,D])
            print ('Zoom out')
        # zoom in
        if key == ord('i'):
            zoomLevel=zoomLevel-0.05
            rows,cols,ch = replaceImg.shape
            pts1 = np.float32([[0, 0],[0,rows],[(cols),(rows)],[cols,0]])
            pts1 = pts1 + np.float32([[-zoomLevel*cols,-zoomLevel*rows],
                                      [-zoomLevel*cols,zoomLevel*rows],
                                      [zoomLevel*cols,zoomLevel*rows],
                                      [zoomLevel*cols,-zoomLevel*rows]])
            print ('Zoom in')
        # new image
        if key == ord('n'):
            if currentImage<len(images)-1:
                currentImage = currentImage+1
                replaceImg = cv2.imread(images[currentImage])
                rows, cols, ch = replaceImg.shape
                pts1 = np.float32([[0, 0], [0, rows], [(cols), (rows)], [cols, 0]])
                pts1 = pts1 + np.float32([[-zoomLevel * cols, -zoomLevel * rows],
                                          [-zoomLevel * cols, zoomLevel * rows],
                                          [zoomLevel * cols, zoomLevel * rows],
                                          [zoomLevel * cols, -zoomLevel * rows]])
                print ('Next image')
            else:
                print ('No more images on the right')
        # previous image
        if key == ord('p'):
            if currentImage>0:
                currentImage=currentImage-1
                replaceImg=cv2.imread(images[currentImage])
                rows, cols, ch = replaceImg.shape
                pts1 = np.float32([[0, 0], [0, rows], [(cols), (rows)], [cols, 0]])
                pts1 = pts1 + np.float32([[-zoomLevel * cols, -zoomLevel * rows],
                                          [-zoomLevel * cols, zoomLevel * rows],
                                          [zoomLevel * cols, zoomLevel * rows],
                                          [zoomLevel * cols, -zoomLevel * rows]])
                print ('Previous image')
            else:
                print ('No more images on the left')
        
cap.release()
cv2.destroyAllWindows()