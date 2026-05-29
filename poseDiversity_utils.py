import math as m
import numpy as np
from gpd_utils import JOINTS, normalizePose

M = 500 #0.5m converted to mm since annotated keypoints are in mm
R = 0.6

#in same order as keypoints
WEIGHTS = np.array([
    5, #head
    1, #neck
    4, #left shoulder
    4, #right shoulder
    1, #left hip
    1, #right hip
    15, #left elbow
    15, #right elbom
    0, #left wrist
    0, #right wrist
    50, #left hand <-- weighted halved since keypoint is hand and not finger tip
    50 #right hand
])

def computeDistance(A, B):

    poseA = normalizePose(A)
    poseB = normalizePose(B)

    jointDists = np.linalg.norm(poseA[i]-poseB[i]) 
    perceptualDist = WEIGHTS * (jointDists**r)
    dist = np.sum(np.minimum(perceptualDist, M))

    return dist