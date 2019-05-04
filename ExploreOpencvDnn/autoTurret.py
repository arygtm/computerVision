
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
import os
import serial

#takes in a servo position in degrees and writes it to the serial port where the Arduino will read it and then move the servo.
def writeServoPos(spd):
    ser.write((str(spd + constants.kServoOffsetDeg) + "\n").encode())

def flywheelControl(selectedIndex):
    if selectedIndex is not None:
        ser.write(("w" + "\n").encode())
        temp = 1
    else:
        ser.write(("s" + "\n").encode())

def getSearchBox(selectedBoxCenter, searchBoxHalfWidth, searchBoxHalfHeight, image_width, image_height):
    searchBoxCenter = (np.min((np.max((selectedBoxCenter[0], searchBoxHalfWidth)),  image_width - searchBoxHalfWidth)), \
    np.min((np.max((selectedBoxCenter[1], searchBoxHalfHeight)),  image_height - searchBoxHalfHeight)) )

    searchBox = np.array([searchBoxCenter[0] - searchBoxHalfWidth, \
    searchBoxCenter[1] - searchBoxHalfHeight, \
    searchBoxCenter[0] + searchBoxHalfWidth, \
    searchBoxCenter[1] + searchBoxHalfHeight])

    return searchBox

#Modifies tracklist
def updateTracksAndDraw(trackList, imageOrig, curBoxes, loopStartTime):
    #Copies bounding boxes from Track objects into np array
    trackedBoxes = np.empty((len(trackList), 4))
    for i in range(len(trackList)):
        trackedBoxes[i, :] = trackList[i].meas['box']

    #Uses linear optimization to match boxes from the previous frame with boxes in the current frame.
    matches = intersection.matchBoxes(trackedBoxes,curBoxes)
    #Draws the previous boxes
    for i in range(trackedBoxes.shape[0]):
        trackedBox = trackedBoxes[i]
        color = constants.colors[-4]
        cv2.rectangle(imageOrig, (int(trackedBox[0]), int(trackedBox[1])), (int(trackedBox[2]), int(trackedBox[3])), color, thickness=1)

    for i in range(curBoxes.shape[0]):
        curBox = curBoxes[i]
        color = constants.colors[0]
        cv2.rectangle(imageOrig, (int(curBox[0]), int(curBox[1])), (int(curBox[2]), int(curBox[3])), color, thickness=4)

    #Loop through all tracks, if there is a match update with match box, else update with None and check if it should be deleted
    #Loop through all detections, if there is a match do nothing, else create new Track and append to tracklist

    #Update loop
    for i in range(len(trackList)):
        if i in matches[0]:
            curBoxMatchIndex = np.where(matches[0] == i)[0][0]
            newMeas = {'captureTime': loopStartTime, 'box': curBoxes[curBoxMatchIndex]}
            trackList[i].update(newMeas)
        else:
            trackList[i].update(None)
    #Delete Loop
    for i in range(len(trackList)-1,0-1, -1):
        if trackList[i].timesUnseenConsecutive > constants.timesUnseenConsecutiveMax:
            #print('POPPING TRACK: ', i)
            trackList.pop(i)

    for i in range(curBoxes.shape[0]):
        if i not in matches[1]:
            initMeas = {'captureTime': loopStartTime, 'box': curBoxes[i]}
            trackList.append(Track(initMeas))

def getSelectionAndPredict(trackList, imageOrig, image_height, prevPrediction):
    selectedIndex = None
    for i in range(len(trackList)):
        if not trackList[i].selected:
            continue

        selectedIndex = i
        curState, guessCovariance = trackList[i].predict(loopStartTime + constants.kalmanPredictionConst)

        if curState is None:
            continue

        cv2.circle(img = imageOrig, center = (int(prevPrediction), int(image_height/2)) , radius = 20 , color = (0, 0, 255), thickness = 4)
        cv2.circle(img = imageOrig, center = (int(prevPrediction), int(image_height/2)) , radius = 1 , color = (0, 0, 255), thickness = 1)

        #cv2.circle(img = image, center = (int(curState[0]), int(480)) , radius = 20 , color = (0, 255, 0), thickness = 4)
        #cv2.circle(img = image, center = (int(curState[0]), int(480)) , radius = 1 , color = (0, 255, 0), thickness = 1)
        #newPrevPrediction = curState[0]
        return (selectedIndex, curState[0])
    return(selectedIndex, prevPrediction)

def selectTarget(trackList):
    bestTrack = {'index': None, 'timesSeenTotal': None}
    for i in range(len(trackList)):
        if bestTrack['timesSeenTotal'] == None or trackList[i].timesSeenTotal > bestTrack['timesSeenTotal']:
            bestTrack = {'index': i, 'timesSeenTotal': trackList[i].timesSeenTotal}
    #Select new target
    if bestTrack['index'] != None and trackList[bestTrack['index']].filter is not None:
        trackList[bestTrack['index']].selected = True

def aimTurretAndDraw(targetXPixel):
    servoPixelOffset = 5520/43 - targetXPixel*23/172
    points = np.array([targetXPixel + servoPixelOffset, 540], dtype = np.float32)
    points = np.reshape(points, (1,1,2))
    undistCenter = cv2.undistortPoints(points, constants.K, constants.dist)[0,0,0]
    cv2.circle(img = imageOrig, center = (int(targetXPixel), int(orig_height/2)) , radius = 20 , color = (0, 255, 0), thickness = 4)

    servoTargetDeg = np.arctan(undistCenter) * 180/np.pi
    writeServoPos(int(90+servoTargetDeg))
    print("targetXPixel", targetXPixel, "undistCenter", undistCenter, "servoTargetDeg",servoTargetDeg)

#Model small is the searchBox model. ModelBig is the whole image model.
modelSmall = cv2.dnn.readNetFromTensorflow('models/MobileNet-SSDLite-v2/frozen_inference_graph.pb','models/MobileNet-SSDLite-v2/ssdlite_mobilenet_v2_coco.pbtxt')
modelBig = cv2.dnn.readNetFromTensorflow('models/MobileNet-SSDLite-v2/frozen_inference_graph.pb','models/MobileNet-SSDLite-v2/ssdlite_mobilenet_v2_coco.pbtxt')
#Starts the camera
cap = cv2.VideoCapture(0)
#Starts a videowriter
outLabeled = cv2.VideoWriter("../../BenchmarkVideos/TrackerLabeled/autoTurretOutLabeled.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 1, (1920,1080))
outOrig = cv2.VideoWriter("../../BenchmarkVideos/TrackerLabeled/autoTurretOutOrig.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 1, (1920,1080))

ser = serial.Serial('/dev/cu.usbmodem14111')#Set this to the actual serial port name

#Variables to store detections
trackList = []
trackedBoxes = np.empty((0,5))#x1, y1, x2, y2, numDetections

prevPrediction = 0

#Logging variables
measurement = np.array([])
imageCaptureTimes = np.array([])
filterState = np.empty((0,2))
filterCovariance = np.empty((0,2,2))
isSelected = np.array([])

selectedIndex = None

startTime = time.time()

#Starts the servo in the center
writeServoPos(90)

prevFireTime = time.time()

while(True):
    r, imageOrig = cap.read()
    if not r:
        continue
    loopStartTime = time.time()
    outOrig.write(imageOrig)
    orig_height, orig_width, _ = imageOrig.shape
    searchBox = None
    curBoxes = None
    networkEndTime = None

    flywheelControl(selectedIndex)
    #runSearchBox()

    curBoxes, networkEndTime = labelImage.findBoxes(modelBig, imageOrig, searchBox, constants.kDetectionThreshold)
    updateTracksAndDraw(trackList, imageOrig, curBoxes, loopStartTime)

    selectedIndex, _ = getSelectionAndPredict(trackList, imageOrig, orig_height, prevPrediction)

    aimTurretAndDraw(960)

    #If no target has been selected then select a new target based on the highest quality detection (whichever has been seen the most)
    if selectedIndex is None:
        selectTarget(trackList)
    else:
        selectedTrack = trackList[selectedIndex]

        #Pickle Data dump
        measurement = np.append(measurement, (selectedTrack.meas['box'][0] + selectedTrack.meas['box'][2])/2)
        imageCaptureTimes = np.append(imageCaptureTimes, selectedTrack.meas['captureTime'])
        filterState = np.vstack((filterState, selectedTrack.filter.prevStateMean))
        filterCovariance = np.concatenate((filterCovariance, np.reshape(selectedTrack.filter.prevStateCovariance, (1,2,2)) ), axis = 0)
        isSelected = np.append(isSelected, 1)

        curState, _ = selectedTrack.predict(loopStartTime + constants.kalmanPredictionConst)
        #aimTurretAndDraw(curState[0])


        #fireTurret by pulsing the solenoid
        if(loopStartTime - prevFireTime > 1):
            ser.write(("f" + "\n").encode())
            prevFireTime = loopStartTime

    #print('time:', networkEndTime - loopStartTime)
    outLabeled.write(imageOrig)
    cv2.imshow('image', cv2.resize(imageOrig, (1280,720))) #Resizing so image fits on screen

    k = cv2.waitKey(1)
    if k == 0xFF & ord("q"):
        ser.write(("s" + "\n").encode())
        break
    elif k == 0xFF & ord("p"):
        ser.write(("s" + "\n").encode())
        pdb.set_trace()

dictionary = {
    'measurement': measurement,
    'imageCaptureTimes': imageCaptureTimes,
    'filterState': filterState,
    'filterCovariance': filterCovariance,
    'isSelected': isSelected
    }

pickleDir = '/Users/arygout/Documents/aaStuff/computerVision/'

pickle.dump(dictionary, open('videoDump.pkl', 'wb'))
ser.write(("s" + "\n").encode())
cv2.waitKey(0)
cv2.destroyAllWindows()

######################################################################################################################################

#Log Zeros
"""
        measurement = np.append(measurement, 0)
        imageCaptureTimes = np.append(imageCaptureTimes, loopStartTime)
        filterState = np.vstack((filterState, np.array([0,0])))
        filterCovariance = np.concatenate((filterCovariance, np.zeros((1,2,2))), axis = 0)
        isSelected = np.append(isSelected,0)"""


#Function won't work. Moved because we are not using it.
def runSearchBox():

    #TODO: Constants. if you get this working move to the constants file
    kSearchBoxHalfWidth = 1000 #TODO: 320
    kSearchBoxHalfHeight = 1000 #TODO: 180

    if selectedIndex is not None: #When a track is currently selected...

        #ser.write(("w" + "\n").encode())

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
        ser.write(("s" + "\n").encode())
