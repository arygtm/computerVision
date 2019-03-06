import cv2
import numpy as np


#Reads in video and captures images when c is pressed. Used to take chessboard images
cap = cv2.VideoCapture(1)

imgCounter = 14

while True:
    ret, frame = cap.read()
    if ret:
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('c'):
            cv2.imwrite('Chessboards/C930e/Original/chessboard' + str(imgCounter) + '.png', frame)
            imgCounter += 1


        cv2.imshow('frame',frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
