
#Base code from: https://heartbeat.fritz.ai/real-time-object-detection-on-raspberry-pi-using-opencv-dnn-98827255fa60

import cv2
import time
import numpy as np
#from imutils import non_max_suppression
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


#Experimenting with different pre-trained models

# Loading model

#Works well 16fps
model0 = cv2.dnn.readNetFromTensorflow('models/MobileNet-SSDLite-v2/frozen_inference_graph.pb','models/MobileNet-SSDLite-v2/ssdlite_mobilenet_v2_coco.pbtxt')

#Works well 14fps
model1 = cv2.dnn.readNetFromTensorflow('models/MobileNet-SSD-v2/frozen_inference_graph.pb','models/MobileNet-SSD-v2/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

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

movieDir = '/Users/arygout/Documents/aaStuff/computerVision/BenchmarkVideos/'

kDetectionThreshold = 0.43

def labelVideo(model,detectionThreshold,frameSkip,movieIn,movieOut):

    cap = cv2.VideoCapture(movieIn)
    out = cv2.VideoWriter(movieOut,cv2.VideoWriter_fourcc('M','J','P','G'), 1, (1280,720))

    frameCounter = -1

    trackedBoxes = np.empty((0,4))

    while(True):
        frameCounter += 1
        r, image = cap.read()
        if frameCounter % frameSkip != 0:
            continue
        if r:
            start_time = time.time()
            image_height, image_width, _ = image.shape

            model.setInput(cv2.dnn.blobFromImage(image, size=(640, 320), swapRB=True))

            output = model.forward()
            #print(output[0,0,:,:])

            end_time = time.time()#Checks the time to label a single frame. Allows for easy comparison of networks.
            print("Elapsed Time:",end_time-start_time)

            #boxes = non_max_suppression(output[0, 0, :, :])
            #print(output[0, 0, :, :].shape)
            allCurBoxes = np.empty((0,5))
            for detection in output[0, 0, :, :]:
                confidence = detection[2]
                class_id = detection[1]
                if confidence > detectionThreshold and class_id == 1:

                    class_name=id_class_name(class_id,classNames)
                    print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
                    box_x = detection[3] * image_width
                    box_y = detection[4] * image_height
                    box_width = detection[5] * image_width
                    box_height = detection[6] * image_height
                    #cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
                    #cv2.putText(image, str(detection[2]) ,(int(box_x), int(box_y)),cv2.FONT_HERSHEY_SIMPLEX,(.001*image_width),(0, 0, 255))
                    allCurBoxes = np.vstack((allCurBoxes, [int(box_x), int(box_y), int(box_width), int(box_height), confidence]))

            nmsOut = np.array(non_max_suppression(allCurBoxes[:,0:4],allCurBoxes[:,4]))
            matches = matchBoxes(trackedBoxes,nmsOut[:,0:4])
            if trackedBoxes.shape[0] != 0:
                for i in range(curBoxes.shape[0]):
                    row = curBoxes[i]
                    cv2.rectangle(image, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (0, 0, 255), thickness=1)
            curBoxes = np.empty((0,4))
            if nmsOut.shape[0] != 0:
                curBoxes = nmsOut[:,0:4]
                probs = nmsOut[:,4]
                for i in range(curBoxes.shape[0]):
                    row = curBoxes[i]
                    cv2.rectangle(image, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (23, 230, 210), thickness=1)
                    #db.set_trace()
                    cv2.putText(image, str(probs[i]) , (int(row[0]),int(row[1])),cv2.FONT_HERSHEY_SIMPLEX,(.001*image_width),(0, 0, 255))
            #if trackedBoxes.shape[0] != 0 and nmsOut.shape[0] != 0:

            trackedBoxes = curBoxes
            """if(range(curBoxes.shape[0] >= 2)):
                for i in range(curBoxes.shape[0]):
                    for j in range(curBoxes.shape[0]):
                        if(j != i):
                            iBox = intersection(curBoxes[i], curBoxes[j])
                            #cv2.rectangle(image, (int(iBox[0]), int(iBox[1])), ((int(iBox[2]-iBox[0])), int(iBox[3]-iBox[1])), (23, 230, 0), thickness=1)
                            if not np.all(iBox == None):
                                cv2.rectangle(image, (int(iBox[0]), int(iBox[1])), ((int(iBox[2])), int(iBox[3])), (255, 0, 0), thickness=1)
                            else:
                                print("detections do not overlap")
                            #cv2.rectangle(image, (int(iBox[0]), int(iBox[1])), ((int(iBox[2]-iBox[0])), int(iBox[3]-iBox[1])), (23, 230, 0), thickness=1)
                            #print(iBox)

                            #cv2.circle(image, (int(iBox[0][0]+iBox[1][0]),int(iBox[0][1]+iBox[1][1])) ,20,(0, 255, 0),thickness = 5)
                            #cv2.circle(image, (int(iBox[0][0]), int(iBox[0][1])) ,20,(0, 255, 0),thickness = 5)
                            #cv2.circle(image, (int(iBox[1][0]), int(iBox[1][1])) ,20,(0, 255, 0),thickness = 5)"""
            out.write(image)
            cv2.imshow('image', image)
        else:
            break
        k = cv2.waitKey(1)

        if k == 0xFF & ord("q"):
            break
        # cv2.imwrite("image_box_text.jpg",image)
        #pdb.set_trace()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for file in os.listdir(movieDir):
    if(file[-3:] == "mov"):
        print(file[0:-4])
        video = os.path.join(movieDir,file)
        labelVideo(model0,kDetectionThreshold,4,video,movieDir+'Labeled/MobileNet-SSDLite-v2/' + file[0:-4] + 'Labeled.avi')
        #labelVideo(model1,kDetectionThreshold,4,video,movieDir+'Labeled/MobileNet-SSD-v2/' + file[0:-4] + 'Labeled.avi')

#pdb.set_trace()
#labelVideo(model0,kDetectionThreshold,5,movieDir+'AryaRunning.mov',movieDir+'Labeled/MobileNet-SSDLite-v2/AryaRunningLabeled.avi')
#labelVideo(model0,kDetectionThreshold,5,movieDir+'AryaRunning.mov',movieDir+'Labeled/MobileNet-SSDLite-v2/AryaRunningLabeled.avi')
