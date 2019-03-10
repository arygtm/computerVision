import pdb
from pykalman import KalmanFilter
import constants

class Track():
    def __init__(self, meas):
        self.meas = meas
        self.timesSeenTotal = 1
        self.timesUnseenConsecutive = 0
        self.filter = None
        self.selected = False

    def predict(self, predTime):
        mean = filter.predict(predTime)
        return mean

    def update(self, newMeas):
        if newMeas is None:
            self.timesUnseenConsecutive += 1
            return

        newX = (newMeas['box'][0] + newMeas['box'][2])/2
        prevX = (meas['box'][0] + meas['box'][2])/2
        if self.filter is None:
            self.filter = OurFilter(newX, prevX, newMeas['captureTime'], meas['captureTime'])

        self.timesSeenTotal += 1
        self.timesUnseenConsecutive = 0
        self.filter.update(newX, newMeas['captureTime'])
        self.meas = newMeas

measurementVariance = 10**2
initStateCovariance = 100**2#TODO: Set this later
allVarianceV = 1.5**2
bigT = 1
avgDistToTarget = 5 #Meters
nominalDt = 0.4
dtForInitCov = nominalDt * 3
class OurFilter():
    def __init__(self, newX, prevX, newCaptureTime, prevCaptureTime):
        _, Q = getTransitionMats(dtForInitCov)

        dt = (newCaptureTime - prevCaptureTime)

        if dt > nominalDt * 3:
            print("dt was huge", dt)
            pdb.set_trace()

        initV = (newX - prevX) / dt
        self.prevStateMean = np.array([newX,initV])
        self.prevStateCovariance = Q
        self.prevStateTime = newCaptureTime
        filter = KalmanFilter()

    def getTransitionMats(dt):
        A = np.array([[1,dt],[0,1]])
        velVariance = allVarianceV * dt/bigT * constants.K[0,0]**2 / avgDistToTarget**2
        posVariance = dt**2/2 * velVariance
        Q = np.diag([posVariance, velVariance])
        return (A,Q)

    def predict(self,predTime):
        A, _ = getTransitionMats(predTime - self.prevStateTime)
        return np.dot(A, self.prevStateMean)

    def update(self,measX,measTime):
        dt = measTime - self.prevStateTime
        A, Q = getTransitionMats(dt)
        self.prevStateMean, self.prevStateCovariance = filter.filter_update( \
            self.prevStateMean,
            self.prevStateCovariance,
            observation = measX,
            transition_matrix = A,
            transition_covariance = Q,
            observation_matrix = np.array([[1,0]]),
            observation_covariance=np.array([[measurementVariance]])
        )
        self.prevStateTime = measTime
