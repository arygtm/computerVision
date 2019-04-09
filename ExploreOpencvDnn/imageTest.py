import cv2
import numpy as np
import aryaNms
import pdb

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
model0 = cv2.dnn.readNetFromTensorflow('models/MobileNet-SSDLite-v2/frozen_inference_graph.pb','models/MobileNet-SSDLite-v2/ssdlite_mobilenet_v2_coco.pbtxt')

#Works well 14fps
model = cv2.dnn.readNetFromTensorflow('models/MobileNet-SSD-v2/frozen_inference_graph.pb','models/MobileNet-SSD-v2/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

movieDir = '/Users/arygout/Documents/aaStuff/BenchmarkVideos/C930e/'

imageOrig = cv2.imread(movieDir + 'AryaWalking.png')

searchBox = np.array([200,0,800,720])

colors = np.array([(255,0,0), (255,128,0), (255,255,0), (128,255,0), (0,255,0), (0,255,128), (0,255,255), (128,255), (0,0,255), (127,0,255), (255,0,255), (255,0,127)])

cv2.rectangle(imageOrig, (int(searchBox[0]), int(searchBox[1])), (int(searchBox[2]), int(searchBox[3])), colors[9], thickness=1)

image = imageOrig[searchBox[1]:searchBox[3],searchBox[0]:searchBox[2],:]

image_height, image_width, _ = image.shape

model.setInput(cv2.dnn.blobFromImage(image, swapRB=True))

output = model.forward()

detectionThreshold = 0.43

allCurBoxes = np.empty((0,4))
allProbs = np.empty((0,1))
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
        #cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
        #cv2.putText(image, str(detection[2]) ,(int(box_x), int(box_y)),cv2.FONT_HERSHEY_SIMPLEX,(.001*image_width),(0, 0, 255))
        allCurBoxes = np.vstack((allCurBoxes, [int(box_x), int(box_y), int(box_width), int(box_height)]))
        allProbs = np.vstack((allProbs, confidence))

curBoxes, curProbs = aryaNms.non_max_suppression(allCurBoxes, allProbs)

for i in range(curBoxes.shape[0]):
    row = np.copy(curBoxes[i])
    rowShift = np.array([searchBox[0], searchBox[1], searchBox[0], searchBox[1]])
    curBoxes[i] += rowShift

for i in range(curBoxes.shape[0]):
    curBox = curBoxes[i]
    color = (255,0,0)#Blue if no match. Cyan if match.
    cv2.rectangle(imageOrig, (int(curBox[0]), int(curBox[1])), (int(curBox[2]), int(curBox[3])), color, thickness=4)

cv2.imshow('image', imageOrig)
cv2.waitKey(500)
pdb.set_trace()
