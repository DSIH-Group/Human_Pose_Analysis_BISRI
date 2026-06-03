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

    jointDists = np.linalg.norm((poseA-poseB),axis=1) 
    perceptualDist = WEIGHTS * (jointDists**R)
    dist = np.sum(np.minimum(perceptualDist, M)) #anything index over M gets capped at M

    return dist

def computePairwiseDistances(framePoses):
    pairwiseDist = []
    
    numPoses = len(framePoses)
    for i in range(numPoses):
        for j in range(i+1, numPoses):
            dist = computeDistance(framePoses[i], framePoses[j])
            pairwiseDist.append(dist)

    return np.array(pairwiseDist)   

def getSummaries(framePoses):
    if len(framePoses) < 2: #if frame only has 1 clinician
        return np.zeros(6)

    #computing pairwise pose distances
    pairwiseDist = computePairwiseDistances(framePoses)

    meanDiff = np.mean(pairwiseDist)
    std = np.std(pairwiseDist)
    maxDist = np.max(pairwiseDist)
    maxDeviation = maxDist-meanDiff
    minDist = np.min(pairwiseDist)
    distRange = maxDist - minDist

    return np.array([])

def testSummary():
    # create test poses  - 4 clinicians in a frame
    pose1 = np.array([
        [550.00, -700.00, 2400.00],  # head
        [550.00, -630.00, 2400.00],  # neck
        [650.00, -580.00, 2400.00],  # left_shoulder
        [450.00, -580.00, 2400.00],  # right_shoulder
        [645.00, -200.00, 2400.00],  # left_hip
        [455.00, -200.00, 2400.00],  # right_hip
        [660.00, -400.00, 2400.00],  # left_elbow
        [440.00, -400.00, 2400.00],  # right_elbow
        [665.00, -250.00, 2400.00],  # left_wrist
        [435.00, -250.00, 2400.00],  # right_wrist
        [668.00, -180.00, 2400.00],  # left_hand
        [432.00, -180.00, 2400.00],  # right_hand
    ])

    pose2 = np.array([
        [500.00, -600.00, 2200.00],  # head
        [510.00, -520.00, 2280.00],  # neck
        [620.00, -480.00, 2350.00],  # left_shoulder
        [400.00, -490.00, 2360.00],  # right_shoulder
        [615.00, -150.00, 2600.00],  # left_hip
        [405.00, -140.00, 2595.00],  # right_hip
        [700.00, -350.00, 2200.00],  # left_elbow
        [300.00, -360.00, 2210.00],  # right_elbow
        [750.00, -200.00, 2050.00],  # left_wrist
        [260.00, -210.00, 2060.00],  # right_wrist
        [780.00, -150.00, 1950.00],  # left_hand
        [230.00, -160.00, 1960.00],  # right_hand
    ])


    pose3 = np.array([
        [580.00, -650.00, 2350.00],  # head
        [575.00, -580.00, 2360.00],  # neck
        [680.00, -540.00, 2370.00],  # left_shoulder
        [470.00, -545.00, 2375.00],  # right_shoulder
        [675.00, -160.00, 2650.00],  # left_hip
        [465.00, -155.00, 2645.00],  # right_hip
        [750.00, -400.00, 2200.00],  # left_elbow
        [480.00, -390.00, 2380.00],  # right_elbow
        [820.00, -280.00, 2050.00],  # left_wrist
        [490.00, -300.00, 2370.00],  # right_wrist
        [870.00, -220.00, 1950.00],  # left_hand
        [495.00, -280.00, 2360.00],  # right_hand
    ])

    pose4 = np.array([
        [560.00, -710.00, 2410.00],  # head
        [558.00, -640.00, 2408.00],  # neck
        [655.00, -585.00, 2405.00],  # left_shoulder
        [455.00, -583.00, 2407.00],  # right_shoulder
        [648.00, -205.00, 2408.00],  # left_hip
        [458.00, -203.00, 2406.00],  # right_hip
        [663.00, -405.00, 2407.00],  # left_elbow
        [443.00, -403.00, 2406.00],  # right_elbow
        [667.00, -253.00, 2406.00],  # left_wrist
        [437.00, -251.00, 2405.00],  # right_wrist
        [670.00, -183.00, 2405.00],  # left_hand
        [434.00, -181.00, 2404.00],  # right_hand
    ])

    framePoses = [pose1, pose2, pose3, pose4]

    # test computeDistance
    print("Testing computeDistance...")
    dist_1_2 = computeDistance(pose1, pose2)
    dist_1_4 = computeDistance(pose1, pose4)
    assert isinstance(dist_1_2, (float, np.floating)), f"Expected float got {type(dist_1_2)}"
    assert dist_1_2 >= 0, "Distance should be non negative"
    assert dist_1_4 < dist_1_2, "Similar poses should have smaller distance than different poses"
    print(f"  pose1 vs pose2 (different): {dist_1_2:.2f}")
    print(f"  pose1 vs pose4 (similar):   {dist_1_4:.2f}")
    print("  computeDistance: OK")

    # test computePairwiseDistances
    print("\nTesting computePairwiseDistances...")
    pairwiseDist = computePairwiseDistances(framePoses)
    expected_pairs = 6  # C(4,2)
    assert pairwiseDist.shape == (expected_pairs,), f"Expected ({expected_pairs},) got {pairwiseDist.shape}"
    assert np.all(pairwiseDist >= 0), "All distances should be non negative"
    print(f"  Number of pairs: {len(pairwiseDist)} (expected {expected_pairs})")
    print(f"  Distances: {pairwiseDist}")
    print("  computePairwiseDistances: OK")

    # test getSummaries
    print("\nTesting getSummaries...")
    summaries = getSummaries(framePoses)
    assert summaries.shape == (6,), f"Expected (6,) got {summaries.shape}"
    assert not np.any(np.isnan(summaries)), "Summaries should not contain NaN"
    assert not np.any(np.isinf(summaries)), "Summaries should not contain inf"

    meanDiff, std, maxDist, maxDeviation, minDist, distRange = summaries
    assert minDist <= meanDiff <= maxDist, "Mean should be between min and max"
    assert std >= 0, "Std should be non negative"
    assert distRange >= 0, "Range should be non negative"
    assert maxDeviation >= 0, "Max deviation should be non negative"
    print(f"  meanDiff:     {meanDiff:.2f}")
    print(f"  std:          {std:.2f}")
    print(f"  maxDist:      {maxDist:.2f}")
    print(f"  maxDeviation: {maxDeviation:.2f}")
    print(f"  minDist:      {minDist:.2f}")
    print(f"  distRange:    {distRange:.2f}")
    print("  getSummaries: OK")

    # test edge case - single pose
    print("\nTesting getSummaries edge case (single pose)...")
    singlePose = getSummaries([pose1])
    assert singlePose.shape == (6,), f"Expected (6,) got {singlePose.shape}"
    assert np.all(singlePose == 0), "Single pose should return zeros"
    print("  getSummaries single pose: OK")

    # test semantic validity - diverse frame vs uniform frame
    print("\nTesting semantic validity...")
    uniformFrame = [pose1, pose4]          # two similar poses
    diverseFrame = [pose1, pose2, pose3]   # three different poses
    uniformSummaries = getSummaries(uniformFrame)
    diverseSummaries = getSummaries(diverseFrame)
    assert diverseSummaries[0] > uniformSummaries[0], "Diverse frame should have higher mean distance"
    print(f"  uniform meanDiff: {uniformSummaries[0]:.2f}")
    print(f"  diverse meanDiff: {diverseSummaries[0]:.2f}")
    print("  Semantic validity: OK")

    print("\nAll diversity tests passed!")

    return 0



if __name__ == "__main__":
    testSummary() #all tests pass