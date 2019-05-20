
#Following code copied verbatim from: https://github.com/jrosebr1/imutils/blob/master/imutils/object_detection.py

# import the necessary packages
import numpy as np
from intersection import intersectionOverUnion
import pdb

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return np.array([])

	# initialize the list of picked indexes
	pick = []

	# if probabilities are provided, sort on them instead
	if probs is not None:
		sort_val = probs
	else:
		sort_val = boxes[:, 4]
	# sort the indexes
	idxs = np.argsort(sort_val.ravel())
	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		overlap = np.zeros((last))
		for ind in range(last):
			boxA = boxes[i, :]
			boxB = boxes[idxs[ind], :]
			overlap[ind] = intersectionOverUnion(boxA, boxB)

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	return boxes[pick, :]
