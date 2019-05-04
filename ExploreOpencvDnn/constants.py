import numpy as np

#Camera variables
kImageWidthPx = 1920
kCameraFOVRads = np.pi/2;
K = np.array( \
    [[1.15092002e+03, 0.00000000e+00, 9.20072938e+02], \
    [0.00000000e+00, 1.14809747e+03, 5.47118233e+02], \
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[ 0.09859361, -0.24158572,  0.00042056, -0.00039583,  0.11073676]])

timesUnseenConsecutiveMax = 5

kDetectionThreshold = 0.43

colors = np.array([(255,0,0), (255,128,0), (255,255,0), (128,255,0), (0,255,0), (0,255,128), (0,255,255), (128,255), (0,0,255), (127,0,255), (255,0,255), (255,0,127)])
#                   Red         Orange      Yellow     Yellow-Green   Green      Blue-Green     Cyan     Light-Blue     Blue    Violet          Magenta   Pink

kServoOffsetDeg = 21

kalmanPredictionConst = 0.3
