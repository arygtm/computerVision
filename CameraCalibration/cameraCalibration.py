import numpy as np
import cv2
import glob
import pdb
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# checkerboard Dimensions
cbrow = 15
cbcol = 10

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('Chessboards/*.png')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (10,15), corners2, ret)
        cv2.imwrite('Chessboards/Labeled/' + fname[12:-4] + 'Original.png', img)
        cv2.imshow('img', img)
        #cv2.waitKey(200)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

cap = cv2.VideoCapture(0)

"""dist = np.array([-4.1802327176423804e-001, 5.0715244063187526e-001, 0, 0, -5.7843597214487474e-001])

while True:
    ret, img = cap.read()
    if ret:


        height, width = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(width,height),1,(width,height))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('c'):
            for i in range(6):
                cv2.imshow('frame',img)
                cv2.waitKey(1000)
                cv2.imshow('frame',dst)
                cv2.waitKey(1000)

        cv2.imshow('frame',img)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()"""



#Reads in the labeled chessboard images and undistorts them by using the camera undistortion  matrix.
labeledImages = glob.glob('Chessboards/Labeled/*.png')

for fname in labeledImages:
    img = cv2.imread(fname)
    height, width = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(width,height),1,(width,height))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    #x,y,w,h = roi
    #dst = dst[y:y+h, x:x+w]

    cv2.imshow('dst', dst)
    cv2.waitKey(200)
    cv2.imwrite('Chessboards/Calibrated/' + fname[20:-11] + 'Undistorted.png',dst)
