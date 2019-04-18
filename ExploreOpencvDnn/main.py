
#Base code from: https://heartbeat.fritz.ai/real-time-object-detection-on-raspberry-pi-using-opencv-dnn-98827255fa60

import cv2
import time
import numpy as np
import intersection
import aryaNms
import pdb
import os
from pykalman import KalmanFilter
import constants
from track import Track
import pickle
import labelImage
from random import randint

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


#Experimenting with different pre-trained models

# Loading model

#Works well 16fps
#model0 = cv2.dnn.readNetFromTensorflow('models/MobileNet-SSDLite-v2/frozen_inference_graph.pb','models/MobileNet-SSDLite-v2/ssdlite_mobilenet_v2_coco.pbtxt')

#Works well 14fps
#model1 = cv2.dnn.readNetFromTensorflow('models/MobileNet-SSD-v2/frozen_inference_graph.pb','models/MobileNet-SSD-v2/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

#Works ok 10fps
#model2 = cv2.dnn.readNetFromTensorflow('models/Inception-SSD-v2/frozen_inference_graph.pb','models/Inception-SSD-v2/ssd_inception_v2_coco_2017_11_17.pbtxt')

#Loads. 2.5fps. No detections
#model3 = cv2.dnn.readNetFromTensorflow('models/faster_rcnn_inception_v2_coco/frozen_inference_graph.pb','models/faster_rcnn_inception_v2_coco/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt')

#Works ok. 0.8fps.
#model4 = cv2.dnn.readNetFromTensorflow('models/MobileNet-SSD-v1-FPN/frozen_inference_graph.pb','models/MobileNet-SSD-v1-FPN/ssd_mobilenet_v1_fpn_coco.pbtxt')

#Loads. 0.5fps. No detections.
#model5 = cv2.dnn.readNetFromTensorflow('models/Faster-RCNN-ResNet-50/frozen_inference_graph.pb','models/Faster-RCNN-ResNet-50/faster_rcnn_resnet50_coco_2018_01_28.pbtxt')

#Loads. 0.18fps. few detections.
#model6 = cv2.dnn.readNetFromTensorflow('models/faster_rcnn_resnet101_kitti/frozen_inference_graph.pb','models/faster_rcnn_resnet101_kitti/faster_rcnn_resnet101_kitti.pbtxt')

movieDir = '/Users/arygout/Documents/aaStuff/BenchmarkVideos/C930e/'

transitionMatrix = np.array( [[1, 0.3], [0, 1]] )
observationMatrix = np.array([0,1])

#transitionCov = np.array([[velstdev, 0],[0, velstdev]])
#observationCov = np.array([pixstdev])

#kf = KalmanFilter(transition_matrices = transitionMatrix, observation_matrices = observationMatrix, transition_covariance = transitionCov, observation_covariance = observationCov)

modelSmall = cv2.dnn.readNetFromTensorflow('models/MobileNet-SSDLite-v2/frozen_inference_graph.pb','models/MobileNet-SSDLite-v2/ssdlite_mobilenet_v2_coco.pbtxt')

modelBig = cv2.dnn.readNetFromTensorflow('models/MobileNet-SSDLite-v2/frozen_inference_graph.pb','models/MobileNet-SSDLite-v2/ssdlite_mobilenet_v2_coco.pbtxt')

def getSearchBox(selectedBoxCenter, searchBoxHalfWidth, searchBoxHalfHeight, image_width, image_height):
    searchBoxCenter = (np.min((np.max((selectedBoxCenter[0], searchBoxHalfWidth)),  image_width - searchBoxHalfWidth)), \
    np.min((np.max((selectedBoxCenter[1], searchBoxHalfHeight)),  image_height - searchBoxHalfHeight)) )

    searchBox = np.array([searchBoxCenter[0] - searchBoxHalfWidth, \
    searchBoxCenter[1] - searchBoxHalfHeight, \
    searchBoxCenter[0] + searchBoxHalfWidth, \
    searchBoxCenter[1] + searchBoxHalfHeight])

    return searchBox

def labelVideo(frameSkip,movieIn,movieOut):

    colors = np.array([(255,0,0), (255,128,0), (255,255,0), (128,255,0), (0,255,0), (0,255,128), (0,255,255), (128,255), (0,0,255), (127,0,255), (255,0,255), (255,0,127)])
    #                   Red         Orange      Yellow     Yellow-Green   Green      Blue-Green     Cyan     Light-Blue     Blue    Violet          Magenta   Pink
    cap = cv2.VideoCapture(movieIn)
    out = cv2.VideoWriter(movieOut,cv2.VideoWriter_fourcc('M','J','P','G'), 1, (1280,720))

    frameCounter = -1

    trackedBoxes = np.empty((0,5))#x1, y1, x2, y2, numDetections

    trackList = []

    prevPrediction = 0

    measurement = np.array([])

    timeStamp = np.array([])

    filterState = np.empty((0,2))

    filterCovariance = np.empty((0,2,2))

    isSelected = np.array([])

    selectedIndex = None

    startTime = time.time()

    kSearchBoxHalfWidth = 320 #TODO: 320
    kSearchBoxHalfHeight = 180 #TODO: 180

    subVals = np.linspace(1,100,100)

    subCounter = 0

    while(True):
        frameCounter += 1
        r, imageOrig = cap.read()
        if frameCounter % frameSkip != 0:
            continue
        if r:
            loopStartTime = time.time()
            orig_height, orig_width, _ = imageOrig.shape

            #Default search box size. Search box keeps this size if no track is selected
            #searchBox = np.array([0,0,orig_width,orig_height])
            searchBox = None

            curBoxes = None

            networkEndTime = None

            if selectedIndex is not None: #When a track is currently selected...


                selectedBox = trackList[selectedIndex].meas['box']

                selectedBoxCenter = (int((selectedBox[0] + selectedBox[2])/2),int((selectedBox[1] + selectedBox[3])/2))

                searchBox = getSearchBox(selectedBoxCenter, kSearchBoxHalfWidth, kSearchBoxHalfHeight, orig_width, orig_height)

                curBoxes, networkEndTime = labelImage.findBoxes(modelSmall, imageOrig, searchBox, constants.kDetectionThreshold)

                #Draws a rectangle to show the search box area
                cv2.rectangle(imageOrig, (int(searchBox[0]), int(searchBox[1])), (int(searchBox[2]), int(searchBox[3])), colors[9], thickness=1)

                #Translate detections by upper left corner coords of search box
                for i in range(curBoxes.shape[0]):
                    row = np.copy(curBoxes[i])
                    rowShift = np.array([searchBox[0], searchBox[1], searchBox[0], searchBox[1]])
                    curBoxes[i] += rowShift
            else:
                curBoxes, networkEndTime = labelImage.findBoxes(modelBig, imageOrig, searchBox, constants.kDetectionThreshold)

            trackedBoxes = np.empty((len(trackList), 4))

            for i in range(len(trackList)):
                trackedBoxes[i, :] = trackList[i].meas['box']

            matches = intersection.matchBoxes(trackedBoxes,curBoxes)
            for i in range(trackedBoxes.shape[0]):
                trackedBox = trackedBoxes[i]
                color = (0, 0, 255) #Red tracked box if no match. Yellow if match
                cv2.rectangle(imageOrig, (int(trackedBox[0]), int(trackedBox[1])), (int(trackedBox[2]), int(trackedBox[3])), color, thickness=1)

            for i in range(curBoxes.shape[0]):
                curBox = curBoxes[i]
                color = (255,0,0)#Blue if no match. Cyan if match.
                cv2.rectangle(imageOrig, (int(curBox[0]), int(curBox[1])), (int(curBox[2]), int(curBox[3])), color, thickness=4)

            #Loop through all tracks, if there is a match update with match box, else update with None and check if it should be deleted
            #Loop through all detections, if there is a match do nothing, else create new Track and append to tracklist
            #Loop through all tracks check

            #Update loop
            for i in range(len(trackList)):
                if i in matches[0]:
                    curBoxMatchIndex = np.where(matches[0] == i)[0][0]
                    newMeas = {'captureTime': networkEndTime, 'box': curBoxes[curBoxMatchIndex]}
                    trackList[i].update(newMeas)
                    #print("updating track:", i, newMeas)

                else:
                    trackList[i].update(None)
                    #print("updating track:", i, "None")
            #Delete Loop
            for i in range(len(trackList)-1,0-1, -1):
                if trackList[i].timesUnseenConsecutive > constants.timesUnseenConsecutiveMax:
                    #print('POPPING TRACK: ', i)
                    trackList.pop(i)

            for i in range(curBoxes.shape[0]):
                if i not in matches[1]:
                    initMeas = {'captureTime': networkEndTime, 'box': curBoxes[i]}
                    trackList.append(Track(initMeas))

            #Checks if a target is selected
            targetSelected = False

            for i in range(len(trackList)):
                if not trackList[i].selected:
                    continue

                targetSelected = True
                selectedIndex = i
                curBoxCenter, guessCovariance = trackList[i].predict(networkEndTime + (networkEndTime - loopStartTime))

                if curBoxCenter is None:
                    continue

                cv2.circle(img = imageOrig, center = (int(prevPrediction), int(480)) , radius = 20 , color = (0, 0, 255), thickness = 4)
                cv2.circle(img = imageOrig, center = (int(prevPrediction), int(480)) , radius = 1 , color = (0, 0, 255), thickness = 1)

                #cv2.circle(img = image, center = (int(curBoxCenter[0]), int(480)) , radius = 20 , color = (0, 255, 0), thickness = 4)
                #cv2.circle(img = image, center = (int(curBoxCenter[0]), int(480)) , radius = 1 , color = (0, 255, 0), thickness = 1)
                prevPrediction = curBoxCenter[0]

            #If no target has been selected then select a new target based on the highest quality detection (whichever has been seen the most)
            if not targetSelected:
                selectedIndex = None
                bestTrack = {'index': None, 'timesSeenTotal': None}
                for i in range(len(trackList)):
                    if bestTrack['timesSeenTotal'] == None or trackList[i].timesSeenTotal > bestTrack['timesSeenTotal']:
                        bestTrack = {'index': i, 'timesSeenTotal': trackList[i].timesSeenTotal}
                #Select new target
                if bestTrack['index'] != None and trackList[bestTrack['index']].filter is not None:
                    trackList[bestTrack['index']].selected = True
                    targetSelected = True

            if selectedIndex != None:
                selectedTrack = trackList[selectedIndex]
                measurement = np.append(measurement, (selectedTrack.meas['box'][0] + selectedTrack.meas['box'][2])/2)
                timeStamp = np.append(timeStamp, selectedTrack.meas['captureTime'])
                filterState = np.vstack((filterState, selectedTrack.filter.prevStateMean))
                filterCovariance = np.concatenate((filterCovariance, np.reshape(selectedTrack.filter.prevStateCovariance, (1,2,2)) ), axis = 0)
                isSelected = np.append(isSelected, 1)
            else:
                measurement = np.append(measurement, 0)
                timeStamp = np.append(timeStamp, networkEndTime)
                filterState = np.vstack((filterState, np.array([0,0])))
                filterCovariance = np.concatenate((filterCovariance, np.zeros((1,2,2))), axis = 0)
                isSelected = np.append(isSelected,0)

            print('time:', networkEndTime - loopStartTime)
            out.write(imageOrig)

            cv2.imshow('image', imageOrig)

        else:
            break
        k = cv2.waitKey(1)

        if k == 0xFF & ord("q"):
            break

        elif k == 0xFF & ord("p"):
            pdb.set_trace()

        endTime = time.time()

    dictionary = {
        'measurement': measurement,
        'timeStamp': timeStamp,
        'filterState': filterState,
        'filterCovariance': filterCovariance,
        'isSelected': isSelected
        }

    pickleDir = '/Users/arygout/Documents/aaStuff/computerVision/'

    #pickle.dump(dictionary, open('videoDump.pkl', 'wb'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for file in os.listdir(movieDir):
    if(file[-3:] == "mov"):
        print(file[0:-4])
        video = os.path.join(movieDir,file)
        movieOutName = movieDir+'Labeled/MobileNet-SSD-v2/' + file[0:-4] + 'Labeled.avi'
        labelVideo(6s,video,movieOutName)

#labelVideo(model0,kDetectionThreshold,5,movieDir+'AryaRunning.mov',movieDir+'Labeled/MobileNet-SSDLite-v2/AryaRunningLabeled.avi')
#labelVideo(model0,kDetectionThreshold,5,movieDir+'AryaRunning.mov',movieDir+'Labeled/MobileNet-SSDLite-v2/AryaRunningLabeled.avi')
#labelVideo(model0,kDetectionThreshold,5,'/Users/arygout/Documents/aaStuff/computerVision/AryaWalking.mov','/Users/arygout/Documents/aaStuff/computerVision/AryaWalkingLabeled.avi')
