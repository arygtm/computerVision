import numpy as np
import pdb
import scipy.optimize

def overlap1D(a1,a2,b1,b2):
    return (a1 <= b1 and a2 >= b1) or (a1 >= b1 and a1 <= b2)

def overlap2D(boxA,boxB):
    #return overlap1D(boxA[0], boxA[0] + boxA[2], boxB[0], boxB[0] + boxB[2]) and overlap1D(boxA[1], boxA[1] + boxA[3], boxB[1], boxB[1] + boxB[3])
    return overlap1D(boxA[0], boxA[2], boxB[0], boxB[2]) and overlap1D(boxA[1], boxA[3], boxB[1], boxB[3])

"""def intersection(boxA, boxB):
  if not overlap2D(boxA, boxB):
      return None
  print(boxA)
  print(boxB)
  boxI = np.array([np.max((boxA[0], boxB[0])),
                   np.max((boxA[1], boxB[1])),
                   np.min((boxA[0] + boxA[2], boxB[0] + boxB[2])),
                   np.min((boxA[1] + boxA[3], boxB[1] + boxB[3]))])
  return boxI"""

#Finds the intersection between two boxes and returns that bounding box. If there is no overlap None is returned
def intersection(boxA, boxB):
  if not overlap2D(boxA, boxB):
      return None
  boxI = np.array([np.max((boxA[0], boxB[0])),
                   np.max((boxA[1], boxB[1])),
                   np.min((boxA[2], boxB[2])),
                   np.min((boxA[3], boxB[3]))])
  return boxI

#Takes in two boxes and returns the area of the intersection divided by the area of the union.
def intersectionOverUnion(boxA,boxB):
    boxI = intersection(boxA,boxB)
    if np.all(boxI) == None:
        return 0
    boxes = np.vstack((boxA,boxB,boxI))
    #Areas A B I
    Areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    return Areas[2] / (Areas[0] + Areas[1] - Areas[2])

#Takes in a vector of trackedBoxes and a vector of curBoxes and matches tracked boxes with curBoxes based on the
#intersection over union. Does this by computing an intersectino over union matrix and optimizing matches using linear_sum_assignment
#If there are no boxes then empty arrays are returned
def matchBoxes(trackedBoxes,curBoxes):
    IoUMatrix = np.empty((trackedBoxes.shape[0],curBoxes.shape[0]))
    for i in range(trackedBoxes.shape[0]):
        for j in range(curBoxes.shape[0]):
            IoU = intersectionOverUnion(trackedBoxes[i,:],curBoxes[j,:])*-1
            if IoU == 0:
                IoU = 1
            IoUMatrix[i][j] = IoU

    if(IoUMatrix.shape[0] > 0) and IoUMatrix.shape[1] > 0:
        return scipy.optimize.linear_sum_assignment(IoUMatrix),IoUMatrix
    else:
        return (np.array([]),np.array([])),IoUMatrix

#Used to run test cases and validate intersection related methods
def runTestCase(boxAList, boxBList, boxITruthList, testCaseName):
  boxA = np.array(boxAList)
  boxB = np.array(boxBList)
  boxITruth = np.array(boxITruthList)

  boxIPredicted = intersection(boxA,boxB)
  if np.all(boxITruth == boxIPredicted):
    print("Test worked: ",testCaseName,)
    return True

  print("Test failed:", testCaseName)
  print("boxA", boxA)
  print("boxB",boxB)
  print("boxITruth", boxITruth)
  print("boxIPredicted", boxIPredicted)
  pdb.set_trace()#Stops the program with pdb if a test case fails
  return False

#Stores all test cases
def unitTestMain():
  runTestCase([3,4,23,20], [9,10,34,30], [9,10,23,20], "Basic")
  runTestCase([3,5,17,22], [1,2,11,15], [3,5,11,15], "ABFlipped")
  runTestCase([3,9,17,27], [10,2,22,19], [10,9,17,19], "Skew")
  runTestCase([10,11,22,24], [1,2,102,104], [10,11,22,24], "TotallyEnclosed")
  runTestCase([2,3,4,5], [100, 101, 10, 11], None, "NoIntersection")
  runTestCase([1,2,10,11], [7,20,15,16], None, "NoIntersectionY")
  runTestCase([5,6,10,11], [20,3,2,8], None, "NoIntersectionX")
  runTestCase([621,286,717,505], [477,276,537,489], None, "NoIntersectionWeird")
  runTestCase([5,10,15,20], [5,15,15,25], [5,15,15,20], "1/3rd Overlap")

unitTestMain()
