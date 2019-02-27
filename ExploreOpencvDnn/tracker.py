#Base code from: https://heartbeat.fritz.ai/real-time-object-detection-on-raspberry-pi-using-opencv-dnn-98827255fa60

import cv2
import time
import numpy as np
from intersection import *
from nms import *
import pdb
import os

# Pretrained classes in the model
classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value

#Works well 16fps
model = cv2.dnn.readNetFromTensorflow('models/MobileNet-SSDLite-v2/frozen_inference_graph.pb','models/MobileNet-SSDLite-v2/ssdlite_mobilenet_v2_coco.pbtxt')

#Works well 14fps
model1 = cv2.dnn.readNetFromTensorflow('models/MobileNet-SSD-v2/frozen_inference_graph.pb','models/MobileNet-SSD-v2/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

detectionThreshold = 0.43

colors = np.array([(255,0,0), (255,128,0), (255,255,0), (128,255,0), (0,255,0), (0,255,128), (0,255,255), (128,255), (0,0,255), (127,0,255), (255,0,255), (255,0,127)])
#                   Red         Orange      Yellow     Yellow-Green   Green      Blue-Green     Cyan     Light-Blue     Blue    Violet          Magenta   Pink
cap = cv2.VideoCapture(0)
#out = cv2.VideoWriter(movieOut,cv2.VideoWriter_fourcc('M','J','P','G'), 1, (1280,720))

frameCounter = -1

frameSkip = 5 #How many frames to skip so the network doesn't lag by processing every single frame

trackedBoxes = np.empty((0,5))#x1, y1, x2, y2, numDetections

while(True):
    frameCounter += 1
    r, image = cap.read()
    if frameCounter % frameSkip != 0:
        continue
    if r:
        start_time = time.time()
        image_height, image_width, _ = image.shape

        model.setInput(cv2.dnn.blobFromImage(image, size=(480, 320), swapRB=True))#Blob is 480x320 which gives decent accuracy and a speed of 10 fps

        output = model.forward()#Finds the detections


        #Looks at all detections and adds all the person detection bounding boxes to allCurBoxes
        allCurBoxes = np.empty((0,5))
        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            class_id = detection[1]
            if confidence > detectionThreshold and class_id == 1:

                class_name=id_class_name(class_id,classNames)
                #print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
                box_x = detection[3] * image_width
                box_y = detection[4] * image_height
                box_width = detection[5] * image_width
                box_height = detection[6] * image_height
                allCurBoxes = np.vstack((allCurBoxes, [int(box_x), int(box_y), int(box_width), int(box_height), confidence]))

        #Uses non-max supression to remove redundant detections, and leave only the detection with the highest network confidence
        nmsOut = np.array(non_max_suppression(allCurBoxes[:,0:4],allCurBoxes[:,4]))
        curBoxes = np.empty((0,4))
        if nmsOut.shape[0] != 0:
            curBoxes = nmsOut[:,0:4]

            #Matches previous detections with current detections and returns an intersection Over Union Matrix
            allMatches, IoUMatrix = matchBoxes(trackedBoxes,curBoxes)

            matches = allMatches

            #Deletes unmatched boxes from matches, leaving only pairs of tracked and current boxes
            for i in range(allMatches[0].shape[0]):
                if IoUMatrix[allMatches[0][i],allMatches[1][i]] == 1: #Positive Sentinel value because matrix entries are negative if there is an intersection
                    matches = (np.delete(allMatches[0],i), np.delete(allMatches[1],i))
            #Loops through trackedBoxes and draws the tracked box with a new color if it has a match
            for i in range(trackedBoxes.shape[0]):
                trackedBox = trackedBoxes[i]
                color = (0, 0, 255) #Red tracked box if no match.
                if i in matches[0]:
                    color = colors[i]#Sets a new color for each detection.
                cv2.rectangle(image, (int(trackedBox[0]), int(trackedBox[1])), (int(trackedBox[2]), int(trackedBox[3])), color, thickness=1)
            #Loops through  curBoxes and draws the tracked box with a new color if it has a match
            for i in range(curBoxes.shape[0]):
                curBox = curBoxes[i]
                color = (255,0,0)#Blue curBox if no match.
                if i in matches[1]:
                    color = colors[i]#Sets a new color for each detection. Same as the tracked one.
                cv2.rectangle(image, (int(curBox[0]), int(curBox[1])), (int(curBox[2]), int(curBox[3])), color, thickness=1)
        trackedBoxes = curBoxes #Saves current boxes to tracked boxes
        #out.write(image)

        #tinyIMG = np.empty((640,480))
        #cv2.resize(image, tinyIMG, tinyIMG)
        cv2.imshow('image', image)
    else:
        break
    k = cv2.waitKey(1)

    if k == 0xFF & ord("q"):
        break

    end_time = time.time()#Checks the time to label a single frame. Allows for easy comparison of networks.
    print("Elapsed Time:",end_time-start_time)


cv2.waitKey(0)
cv2.destroyAllWindows()
