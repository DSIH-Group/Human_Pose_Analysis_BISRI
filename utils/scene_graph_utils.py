import numpy as np
from gpd_utils import JOINTS

def computeTableSummaries(framePoses):
    
    #clinician distribution
    numClinicians = len(framePoses)

    cliniciansLeft = 0 #clinicians to left of table

    for pose in framePoses:
        leftHip = pose[JOINTS['left_hip']]
        rightHip = pose[JOINTS['right_hip']]
        midHip = (leftHip + rightHip)/2

        if midHip[0] < 0: #x coordinate
            cliniciansLeft+=1

    
    leftRightRatio = cliniciansLeft/numClinicians


    #distance summaries 
    frameDists = [] #min joint distance from poses to table
    for pose in framePoses:
        jointDists = np.linalg.norm(pose, axis=1)
        dist = min(jointDists) 
        frameDists.append(dist)
    
    frameDists = np.array(frameDists)
    activeClinicians = len(frameDists[frameDists <= 300])  #clinicians within active zone
    meanTableDist = np.mean(frameDists)
    minTableDist = np.min(frameDists)
    

    return np.array([activeClinicians, meanTableDist, leftRightRatio, minTableDist])

def computeInterPersonDistance(poseA, poseB):
    jointDists = []

    #computing joint distances between all pairs of distances
    for i in range(len(poseA)):
        for j in range(len(poseB)):
            dist = np.linalg.norm((poseA[i]-poseB[j])) 
            jointDists.append(dist)

    return np.mean(jointDists)

def computeInterPersonSummaries(framePoses):
    numClinicians = len(framePoses)

    if numClinicians < 2: #median requires 1 pair
        return np.zeros(4)
    
    pairwiseDists = []
    for i in range(numClinicians):
        for j in range(i+1, numClinicians):
            dist = computeInterPersonDistance(framePoses[i], framePoses[j])
            pairwiseDists.append(dist)
    
    # medianDist = np.median(pairwiseDists)

    #added interperson features (for efa)
    meanDist = np.mean(pairwiseDists)
    std = np.std(pairwiseDists)
    minDist = np.min(pairwiseDists)
    maxDist = np.max(pairwiseDists)

    
    return np.array([meanDist, std, minDist, maxDist])

def computeSceneGraphSummary(framePoses):
    numClinicians = len(framePoses)

    if numClinicians == 0:
        return np.zeros(9)
    
    tableDistSummary = computeTableSummaries(framePoses)
    interPersonSummary = computeInterPersonSummaries(framePoses)
    
    return np.concatenate([[numClinicians], tableDistSummary, interPersonSummary])

def testTableSummaries(frames,labels):
    print("\nTesting computeTableSummaries...")

    # test 1 - output shape
    summary = computeTableSummaries(frames[0])
    assert summary.shape == (5,), f"Expected (5,) got {summary.shape}"
    print("  test 1 - output shape: OK")

    # test 2 - no NaN or inf
    assert not np.any(np.isnan(summary)), "Should not contain NaN"
    assert not np.any(np.isinf(summary)), "Should not contain inf"
    print("  test 2 - no NaN or inf: OK")

    # test 3 - single active clinician
    # frame1 has one clinician close to table (z=10-100)
    summary_f1 = computeTableSummaries(frames[0])
    activeClinicians, meanDist, leftRight, frontBack, minDist = summary_f1
    assert activeClinicians == 1, f"Expected 1 active clinician got {activeClinicians}"
    assert minDist <= 300, f"Min distance should be <= 300 got {minDist}"
    print(f"  test 3 - single active clinician: OK (activeClinicians={activeClinicians}, minDist={minDist:.2f})")

    # test 4 - all clinicians observer zone
    # frame6 has all clinicians at z=1500
    summary_f6 = computeTableSummaries(frames[5])
    activeClinicians_f6 = summary_f6[0]
    assert activeClinicians_f6 == 0, f"Expected 0 active clinicians got {activeClinicians_f6}"
    print(f"  test 4 - all clinicians observer zone active count=0: OK")

    # test 5 - all clinicians active zone
    # frame7 has all clinicians at z=5-50
    summary_f7 = computeTableSummaries(frames[6])
    activeClinicians_f7 = summary_f7[0]
    assert activeClinicians_f7 == len(frames[6]), \
        f"Expected {len(frames[6])} active clinicians got {activeClinicians_f7}"
    print(f"  test 5 - all clinicians active zone: OK (activeClinicians={activeClinicians_f7})")

    # test 6 - left/right ratio all left
    # frame5 has all clinicians with negative x coordinates
    summary_f5 = computeTableSummaries(frames[4])
    leftRight_f5 = summary_f5[2]
    assert np.isclose(leftRight_f5, 1.0), \
        f"Expected left/right ratio 1.0 got {leftRight_f5}"
    print(f"  test 6 - all clinicians left side ratio=1.0: OK")

    # test 7 - left/right ratio balanced
    # frame2 has one clinician each side
    summary_f2 = computeTableSummaries(frames[1])
    leftRight_f2 = summary_f2[2]
    assert np.isclose(leftRight_f2, 0.5), \
        f"Expected left/right ratio 0.5 got {leftRight_f2}"
    print(f"  test 7 - balanced left/right ratio=0.5: OK")

    # test 8 - front/back ratio balanced
    # frame8 has one clinician front (z=-200) one back (z=200)
    summary_f8 = computeTableSummaries(frames[7])
    frontBack_f8 = summary_f8[3]
    assert np.isclose(frontBack_f8, 0.5), \
        f"Expected front/back ratio 0.5 got {frontBack_f8}"
    print(f"  test 8 - balanced front/back ratio=0.5: OK")

    # test 9 - single observer clinician far from table
    # frame9 has clinician at z=1500
    summary_f9 = computeTableSummaries(frames[8])
    minDist_f9 = summary_f9[4]
    assert minDist_f9 > 300, \
        f"Observer clinician min distance should be > 300 got {minDist_f9:.2f}"
    print(f"  test 9 - observer clinician far from table: OK (minDist={minDist_f9:.2f})")

    # test 10 - mean distance active < mean distance observer
    # frame7 all active, frame6 all observers
    meanDist_active = computeTableSummaries(frames[6])[1]
    meanDist_observer = computeTableSummaries(frames[5])[1]
    assert meanDist_active < meanDist_observer, \
        f"Active frame mean dist {meanDist_active:.2f} should be < observer {meanDist_observer:.2f}"
    print(f"  test 10 - active mean dist < observer mean dist: OK")

    # test 11 - ratios always between 0 and 1
    for frame, label in zip(frames, labels):
        s = computeTableSummaries(frame)
        assert 0 <= s[2] <= 1, f"Left/right ratio out of range for {label}"
        assert 0 <= s[3] <= 1, f"Front/back ratio out of range for {label}"
    print("  test 11 - ratios always between 0 and 1: OK")

    # test 12 - active clinicians never exceeds total clinicians
    for frame, label in zip(frames, labels):
        s = computeTableSummaries(frame)
        assert s[0] <= len(frame), \
            f"Active clinicians {s[0]} exceeds total {len(frame)} for {label}"
    print("  test 12 - active clinicians never exceeds total: OK")

    # test 13 - min distance always <= mean distance
    for frame, label in zip(frames, labels):
        s = computeTableSummaries(frame)
        assert s[4] <= s[1], \
            f"Min dist {s[4]:.2f} should be <= mean dist {s[1]:.2f} for {label}"
    print("  test 13 - min distance <= mean distance: OK")

    print("All computeTableSummaries tests passed!")
    
    return 0

def testInterPersonDist(frames):
    print("Testing computeInterPersonDistance...")

    # test 1 - output is scalar
    dist = computeInterPersonDistance(frames[1][0], frames[1][1])
    assert isinstance(dist, (float, np.floating)), f"Expected float got {type(dist)}"
    print("  test 1 - output is scalar: OK")

    # test 2 - non negative
    assert dist >= 0, "Distance should be non negative"
    print("  test 2 - non negative: OK")

    # test 3 - no NaN or inf
    assert not np.isnan(dist), "Should not be NaN"
    assert not np.isinf(dist), "Should not be inf"
    print("  test 3 - no NaN or inf: OK")

    # test 4 - same pose should give smaller distance than different poses
    dist_same = computeInterPersonDistance(frames[1][0], frames[1][0])
    dist_diff = computeInterPersonDistance(frames[1][0], frames[1][1])
    assert dist_same < dist_diff, \
        f"Same pose should give smaller distance than different poses got same={dist_same:.2f} diff={dist_diff:.2f}"
    print("  test 4 - same pose gives smaller distance than different pose: OK")

    # test 5 - symmetry
    # distance from A to B should equal distance from B to A
    dist_ab = computeInterPersonDistance(frames[1][0], frames[1][1])
    dist_ba = computeInterPersonDistance(frames[1][1], frames[1][0])
    assert np.isclose(dist_ab, dist_ba), \
        f"Distance should be symmetric got {dist_ab:.2f} and {dist_ba:.2f}"
    print("  test 5 - symmetry: OK")

    # test 6 - closer poses should give smaller distance
    # frames[1] has two clinicians on opposite sides (~600mm apart)
    # frame3 has two active clinicians closer together (~400mm apart)
    dist_far = computeInterPersonDistance(frames[1][0], frames[1][1])
    dist_close = computeInterPersonDistance(frames[2][0], frames[2][1])
    assert dist_close < dist_far, \
        f"Closer poses should give smaller distance got close={dist_close:.2f} far={dist_far:.2f}"
    print(f"  test 6 - closer poses give smaller distance: OK (close={dist_close:.2f}, far={dist_far:.2f})")

    # test 7 - total number of joint pairs should be 12x12=144
    # verify by checking a known case manually
    poseA = frames[1][0]
    poseB = frames[1][1]
    manual_dists = []
    for i in range(len(poseA)):
        for j in range(len(poseB)):
            manual_dists.append(np.linalg.norm(poseA[i] - poseB[j]))
    assert len(manual_dists) == 144, f"Expected 144 pairs got {len(manual_dists)}"
    assert np.isclose(np.mean(manual_dists), computeInterPersonDistance(poseA, poseB)), \
        "Manual computation should match function output"
    print("  test 7 - 144 joint pairs computed correctly: OK")

    # test 8 - all zeros poses
    zero_pose = np.zeros((12, 3))
    dist_zero = computeInterPersonDistance(zero_pose, zero_pose)
    assert np.isclose(dist_zero, 0.0), "Zero poses should give distance 0"
    print("  test 8 - all zeros poses: OK")

    # test 9 - active clinicians closer than observers
    # compare active pair (frame3[0], frame3[1]) vs active-observer pair (frame3[0], frame3[2])
    dist_active_pair = computeInterPersonDistance(frames[2][0], frames[2][1])
    dist_active_observer = computeInterPersonDistance(frames[2][0], frames[2][2])
    assert dist_active_pair < dist_active_observer, \
        f"Active pair should be closer than active-observer pair"
    print(f"  test 9 - active pair closer than-observer: OK")

    print("\nAll computeInterPersonDistance tests passed!")

def testInterPersonSummaries(frames,labels):
    print("\nTesting computeInterPersonSummaries...")

    # test 1 - output shape single clinician
    summary_single = computeInterPersonSummaries(frames[0])
    assert summary_single.shape == (1,), f"Expected (1,) got {summary_single.shape}"
    print("  test 1 - output shape single clinician: OK")

    # test 2 - single clinician returns zeros
    assert np.all(summary_single == 0), "Single clinician should return zeros"
    print("  test 2 - single clinician returns zeros: OK")

    # test 3 - output shape two clinicians
    summary_two = computeInterPersonSummaries(frames[1])
    assert summary_two.shape == (1,), f"Expected (1,) got {summary_two.shape}"
    print("  test 3 - output shape two clinicians: OK")

    # test 4 - no NaN or inf
    assert not np.any(np.isnan(summary_two)), "Should not contain NaN"
    assert not np.any(np.isinf(summary_two)), "Should not contain inf"
    print("  test 4 - no NaN or inf: OK")

    # test 5 - non negative median
    assert summary_two[0] >= 0, "Median should be non negative"
    print("  test 5 - non negative median: OK")

    # test 6 - closer clinicians give smaller median
    # frame2 has clinicians ~600mm apart
    # frame3 active clinicians ~400mm apart
    summary_far = computeInterPersonSummaries(frames[1])
    summary_close = computeInterPersonSummaries([frames[2][0], frames[2][1]])
    assert summary_close[0] < summary_far[0], \
        f"Closer clinicians should give smaller median got close={summary_close[0]:.2f} far={summary_far[0]:.2f}"
    print(f"  test 6 - closer clinicians give smaller median: OK")

    # test 7 - observer increases median distance
    # two active clinicians vs two active + one observer
    summary_active_only = computeInterPersonSummaries([frames[2][0], frames[2][1]])
    summary_with_observer = computeInterPersonSummaries(frames[2])
    assert summary_with_observer[0] > summary_active_only[0], \
        f"Adding observer should increase median got active={summary_active_only[0]:.2f} with_observer={summary_with_observer[0]:.2f}"
    print(f"  test 7 - observer increases median distance: OK")

    # test 8 - three clinicians gives 3 pairs
    # verify by checking median manually
    pairwise = [
        computeInterPersonDistance(frames[2][0], frames[2][1]),
        computeInterPersonDistance(frames[2][0], frames[2][2]),
        computeInterPersonDistance(frames[2][1], frames[2][2]),
    ]
    expected_median = np.median(pairwise)
    assert np.isclose(summary_with_observer[0], expected_median), \
        f"Expected median {expected_median:.2f} got {summary_with_observer[0]:.2f}"
    print("  test 8 - three clinicians median computed correctly: OK")

    # test 9 - all frames return correct shape
    for frame, label in zip(frames, labels):
        s = computeInterPersonSummaries(frame)
        assert s.shape == (1,), f"Expected (1,) got {s.shape} for {label}"
        assert not np.any(np.isnan(s)), f"NaN found for {label}"
        assert not np.any(np.isinf(s)), f"Inf found for {label}"
    print("  test 9 - all frames return correct shape: OK")

    print("\nAll computeInterPersonSummaries tests passed!")


def testGraphSummaries(frames,labels):
    print("Testing computeSceneGraphSummary...")

    # test 1 - output shape
    summary = computeSceneGraphSummary(frames[0])
    assert summary.shape == (7,), f"Expected (7,) got {summary.shape}"
    print("  test 1 - output shape: OK")

    # test 2 - empty frame returns zeros
    summary_empty = computeSceneGraphSummary([])
    assert summary_empty.shape == (7,), f"Expected (7,) got {summary_empty.shape}"
    assert np.all(summary_empty == 0), "Empty frame should return zeros"
    print("  test 2 - empty frame returns zeros: OK")

    # test 3 - no NaN or inf
    assert not np.any(np.isnan(summary)), "Should not contain NaN"
    assert not np.any(np.isinf(summary)), "Should not contain inf"
    print("  test 3 - no NaN or inf: OK")

    # test 5 - num clinicians correct
    for frame, label in zip(frames, labels):
        s = computeSceneGraphSummary(frame)
        assert s[0] == len(frame), \
            f"Expected {len(frame)} clinicians got {s[0]} for {label}"
    print("  test 5 - num clinicians correct: OK")

    # test 6 - active clinicians never exceeds total clinicians
    for frame, label in zip(frames, labels):
        s = computeSceneGraphSummary(frame)
        assert s[1] <= s[0], \
            f"Active clinicians {s[1]} exceeds total {s[0]} for {label}"
    print("  test 6 - active clinicians never exceeds total: OK")

    # test 7 - all active frame has more active clinicians than all observer frame
    summary_active = computeSceneGraphSummary(frames[6])
    summary_observer = computeSceneGraphSummary(frames[5])
    assert summary_active[1] > summary_observer[1], \
        f"Active frame should have more active clinicians than observer frame"
    print(f"  test 7 - active frame has more active clinicians: OK")

    # test 8 - mean table distance active < mean table distance observer
    assert summary_active[2] < summary_observer[2], \
        f"Active frame mean dist {summary_active[2]:.2f} should be < observer {summary_observer[2]:.2f}"
    print("  test 8 - active mean table dist < observer mean table dist: OK")

    # test 9 - ratios always between 0 and 1
    for frame, label in zip(frames, labels):
        s = computeSceneGraphSummary(frame)
        assert 0 <= s[3] <= 1, f"Left/right ratio out of range for {label}"
        assert 0 <= s[4] <= 1, f"Front/back ratio out of range for {label}"
    print("  test 9 - ratios always between 0 and 1: OK")

    # test 10 - min table distance <= mean table distance
    for frame, label in zip(frames, labels):
        s = computeSceneGraphSummary(frame)
        assert s[5] <= s[2], \
            f"Min dist {s[5]:.2f} should be <= mean dist {s[2]:.2f} for {label}"
    print("  test 10 - min table distance <= mean table distance: OK")

    # test 11 - single clinician inter person distance is zero
    summary_single = computeSceneGraphSummary(frames[0])
    assert summary_single[6] == 0, \
        f"Single clinician inter person distance should be 0 got {summary_single[6]}"
    print("  test 11 - single clinician inter person distance is zero: OK")

    # test 12 - active clinicians closer gives smaller inter person distance
    summary_close = computeSceneGraphSummary(frames[6])   # all active close to table
    summary_far = computeSceneGraphSummary(frames[5])     # all observers far from table
    assert summary_close[6] < summary_far[6], \
        f"Active clinicians should have smaller inter person distance got close={summary_close[6]:.2f} far={summary_far[6]:.2f}"
    print("  test 12 - active clinicians have smaller inter person distance: OK")

    # test 13 - all frames return valid output
    for frame, label in zip(frames, labels):
        s = computeSceneGraphSummary(frame)
        assert s.shape == (7,), f"Expected (7,) got {s.shape} for {label}"
        assert not np.any(np.isnan(s)), f"NaN found for {label}"
        assert not np.any(np.isinf(s)), f"Inf found for {label}"
    print("  test 13 - all frames return valid output: OK")

    print("\nAll computeSceneGraphSummary tests passed!")


def testSceneGraph():
    # test poses at different positions around table
    # Frame 1 - single active clinician leaning over table
    frame1 = [
        np.array([
            [ -100, -700,  100],  # head
            [ -100, -630,  100],  # neck
            [ -200, -580,  100],  # left_shoulder
            [    0, -580,  100],  # right_shoulder
            [ -200, -200,  100],  # left_hip
            [    0, -200,  100],  # right_hip
            [ -250, -350,   50],  # left_elbow
            [   50, -350,   50],  # right_elbow
            [ -280, -150,   20],  # left_wrist
            [   80, -150,   20],  # right_wrist
            [ -300,  -50,   10],  # left_hand
            [  100,  -50,   10],  # right_hand
        ])
    ]

    # Frame 2 - two clinicians on opposite sides of table
    frame2 = [
        np.array([
            [ -300, -700,  100],  # head
            [ -300, -630,  100],  # neck
            [ -400, -580,  100],  # left_shoulder
            [ -200, -580,  100],  # right_shoulder
            [ -400, -200,  100],  # left_hip
            [ -200, -200,  100],  # right_hip
            [ -420, -350,   50],  # left_elbow
            [ -180, -350,   50],  # right_elbow
            [ -440, -150,   20],  # left_wrist
            [ -160, -150,   20],  # right_wrist
            [ -460,  -50,   10],  # left_hand
            [ -140,  -50,   10],  # right_hand
        ]),
        np.array([
            [  300, -700,  100],  # head
            [  300, -630,  100],  # neck
            [  200, -580,  100],  # left_shoulder
            [  400, -580,  100],  # right_shoulder
            [  200, -200,  100],  # left_hip
            [  400, -200,  100],  # right_hip
            [  180, -350,   50],  # left_elbow
            [  420, -350,   50],  # right_elbow
            [  160, -150,   20],  # left_wrist
            [  440, -150,   20],  # right_wrist
            [  140,  -50,   10],  # left_hand
            [  460,  -50,   10],  # right_hand
        ]),
    ]

    # Frame 3 - three clinicians, two active one observer
    frame3 = [
        np.array([
            [ -200, -700,  100],  # head        (active left)
            [ -200, -630,  100],  # neck
            [ -300, -580,  100],  # left_shoulder
            [ -100, -580,  100],  # right_shoulder
            [ -300, -200,  100],  # left_hip
            [ -100, -200,  100],  # right_hip
            [ -320, -350,   50],  # left_elbow
            [  -80, -350,   50],  # right_elbow
            [ -340, -150,   20],  # left_wrist
            [  -60, -150,   20],  # right_wrist
            [ -360,  -50,   10],  # left_hand
            [  -40,  -50,   10],  # right_hand
        ]),
        np.array([
            [  200, -700,  100],  # head        (active right)
            [  200, -630,  100],  # neck
            [  100, -580,  100],  # left_shoulder
            [  300, -580,  100],  # right_shoulder
            [  100, -200,  100],  # left_hip
            [  300, -200,  100],  # right_hip
            [   80, -350,   50],  # left_elbow
            [  320, -350,   50],  # right_elbow
            [   60, -150,   20],  # left_wrist
            [  340, -150,   20],  # right_wrist
            [   40,  -50,   10],  # left_hand
            [  360,  -50,   10],  # right_hand
        ]),
        np.array([
            [    0, -700, 1500],  # head        (observer)
            [    0, -630, 1500],  # neck
            [ -100, -580, 1500],  # left_shoulder
            [  100, -580, 1500],  # right_shoulder
            [ -100, -200, 1500],  # left_hip
            [  100, -200, 1500],  # right_hip
            [ -120, -400, 1500],  # left_elbow
            [  120, -400, 1500],  # right_elbow
            [ -130, -250, 1500],  # left_wrist
            [  130, -250, 1500],  # right_wrist
            [ -140, -180, 1500],  # left_hand
            [  140, -180, 1500],  # right_hand
        ]),
    ]

    # Frame 4 - four clinicians, mixed positions
    frame4 = [
        np.array([
            [ -200, -700,  100],  # head        (active left front)
            [ -200, -630,  100],  # neck
            [ -300, -580,  100],  # left_shoulder
            [ -100, -580,  100],  # right_shoulder
            [ -300, -200,  100],  # left_hip
            [ -100, -200,  100],  # right_hip
            [ -320, -350,   50],  # left_elbow
            [  -80, -350,   50],  # right_elbow
            [ -340, -150,   20],  # left_wrist
            [  -60, -150,   20],  # right_wrist
            [ -360,  -50,   10],  # left_hand
            [  -40,  -50,   10],  # right_hand
        ]),
        np.array([
            [  200, -700,  100],  # head        (active right front)
            [  200, -630,  100],  # neck
            [  100, -580,  100],  # left_shoulder
            [  300, -580,  100],  # right_shoulder
            [  100, -200,  100],  # left_hip
            [  300, -200,  100],  # right_hip
            [   80, -350,   50],  # left_elbow
            [  320, -350,   50],  # right_elbow
            [   60, -150,   20],  # left_wrist
            [  340, -150,   20],  # right_wrist
            [   40,  -50,   10],  # left_hand
            [  360,  -50,   10],  # right_hand
        ]),
        np.array([
            [ -200, -700, 1500],  # head        (observer left back)
            [ -200, -630, 1500],  # neck
            [ -300, -580, 1500],  # left_shoulder
            [ -100, -580, 1500],  # right_shoulder
            [ -300, -200, 1500],  # left_hip
            [ -100, -200, 1500],  # right_hip
            [ -320, -400, 1500],  # left_elbow
            [  -80, -400, 1500],  # right_elbow
            [ -330, -250, 1500],  # left_wrist
            [  -70, -250, 1500],  # right_wrist
            [ -340, -180, 1500],  # left_hand
            [  -60, -180, 1500],  # right_hand
        ]),
        np.array([
            [  200, -700, 1500],  # head        (observer right back)
            [  200, -630, 1500],  # neck
            [  100, -580, 1500],  # left_shoulder
            [  300, -580, 1500],  # right_shoulder
            [  100, -200, 1500],  # left_hip
            [  300, -200, 1500],  # right_hip
            [   80, -400, 1500],  # left_elbow
            [  320, -400, 1500],  # right_elbow
            [   70, -250, 1500],  # left_wrist
            [  330, -250, 1500],  # right_wrist
            [   60, -180, 1500],  # left_hand
            [  340, -180, 1500],  # right_hand
        ]),
    ]

    # Frame 5 - all clinicians on same side (left)
    frame5 = [
        np.array([
            [ -100, -700,  100],  # head
            [ -100, -630,  100],  # neck
            [ -200, -580,  100],  # left_shoulder
            [    0, -580,  100],  # right_shoulder
            [ -200, -200,  100],  # left_hip
            [    0, -200,  100],  # right_hip
            [ -220, -350,   50],  # left_elbow
            [  -20, -350,   50],  # right_elbow
            [ -230, -150,   20],  # left_wrist
            [  -10, -150,   20],  # right_wrist
            [ -240,  -50,   10],  # left_hand
            [   -5,  -50,   10],  # right_hand
        ]),
        np.array([
            [ -300, -700,  100],  # head
            [ -300, -630,  100],  # neck
            [ -400, -580,  100],  # left_shoulder
            [ -200, -580,  100],  # right_shoulder
            [ -400, -200,  100],  # left_hip
            [ -200, -200,  100],  # right_hip
            [ -420, -350,   50],  # left_elbow
            [ -180, -350,   50],  # right_elbow
            [ -440, -150,   20],  # left_wrist
            [ -160, -150,   20],  # right_wrist
            [ -460,  -50,   10],  # left_hand
            [ -140,  -50,   10],  # right_hand
        ]),
    ]

    # Frame 6 - all clinicians in observer zone
    frame6 = [
        np.array([
            [ -100, -700, 1500],  # head
            [ -100, -630, 1500],  # neck
            [ -200, -580, 1500],  # left_shoulder
            [    0, -580, 1500],  # right_shoulder
            [ -200, -200, 1500],  # left_hip
            [    0, -200, 1500],  # right_hip
            [ -220, -400, 1500],  # left_elbow
            [  -20, -400, 1500],  # right_elbow
            [ -230, -250, 1500],  # left_wrist
            [  -10, -250, 1500],  # right_wrist
            [ -240, -180, 1500],  # left_hand
            [   -5, -180, 1500],  # right_hand
        ]),
        np.array([
            [  100, -700, 1500],  # head
            [  100, -630, 1500],  # neck
            [    0, -580, 1500],  # left_shoulder
            [  200, -580, 1500],  # right_shoulder
            [    0, -200, 1500],  # left_hip
            [  200, -200, 1500],  # right_hip
            [  -20, -400, 1500],  # left_elbow
            [  220, -400, 1500],  # right_elbow
            [  -30, -250, 1500],  # left_wrist
            [  230, -250, 1500],  # right_wrist
            [  -40, -180, 1500],  # left_hand
            [  240, -180, 1500],  # right_hand
        ]),
    ]

    # Frame 7 - all clinicians in active zone
    frame7 = [
        np.array([
            [ -100, -700,   50],  # head
            [ -100, -630,   50],  # neck
            [ -200, -580,   50],  # left_shoulder
            [    0, -580,   50],  # right_shoulder
            [ -200, -200,   50],  # left_hip
            [    0, -200,   50],  # right_hip
            [ -220, -350,   30],  # left_elbow
            [  -20, -350,   30],  # right_elbow
            [ -230, -150,   10],  # left_wrist
            [  -10, -150,   10],  # right_wrist
            [ -240,  -50,    5],  # left_hand
            [   -5,  -50,    5],  # right_hand
        ]),
        np.array([
            [  100, -700,   50],  # head
            [  100, -630,   50],  # neck
            [    0, -580,   50],  # left_shoulder
            [  200, -580,   50],  # right_shoulder
            [    0, -200,   50],  # left_hip
            [  200, -200,   50],  # right_hip
            [  -20, -350,   30],  # left_elbow
            [  220, -350,   30],  # right_elbow
            [  -30, -150,   10],  # left_wrist
            [  230, -150,   10],  # right_wrist
            [  -40,  -50,    5],  # left_hand
            [  240,  -50,    5],  # right_hand
        ]),
    ]

    # Frame 8 - clinicians evenly split front/back
    frame8 = [
        np.array([
            [ -100, -700, -200],  # head        (front)
            [ -100, -630, -200],  # neck
            [ -200, -580, -200],  # left_shoulder
            [    0, -580, -200],  # right_shoulder
            [ -200, -200, -200],  # left_hip
            [    0, -200, -200],  # right_hip
            [ -220, -350, -200],  # left_elbow
            [  -20, -350, -200],  # right_elbow
            [ -230, -150, -200],  # left_wrist
            [  -10, -150, -200],  # right_wrist
            [ -240,  -50, -200],  # left_hand
            [   -5,  -50, -200],  # right_hand
        ]),
        np.array([
            [  100, -700,  200],  # head        (back)
            [  100, -630,  200],  # neck
            [    0, -580,  200],  # left_shoulder
            [  200, -580,  200],  # right_shoulder
            [    0, -200,  200],  # left_hip
            [  200, -200,  200],  # right_hip
            [  -20, -350,  200],  # left_elbow
            [  220, -350,  200],  # right_elbow
            [  -30, -150,  200],  # left_wrist
            [  230, -150,  200],  # right_wrist
            [  -40,  -50,  200],  # left_hand
            [  240,  -50,  200],  # right_hand
        ]),
    ]

    # Frame 9 - single clinician in observer zone
    frame9 = [
        np.array([
            [    0, -700, 1500],  # head
            [    0, -630, 1500],  # neck
            [ -100, -580, 1500],  # left_shoulder
            [  100, -580, 1500],  # right_shoulder
            [ -100, -200, 1500],  # left_hip
            [  100, -200, 1500],  # right_hip
            [ -120, -400, 1500],  # left_elbow
            [  120, -400, 1500],  # right_elbow
            [ -130, -250, 1500],  # left_wrist
            [  130, -250, 1500],  # right_wrist
            [ -140, -180, 1500],  # left_hand
            [  140, -180, 1500],  # right_hand
        ]),
    ]

    frames = [frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9]
    labels = [
        'single active clinician',
        'two clinicians opposite sides',
        'three clinicians two active one observer',
        'four clinicians two active two observers',
        'all clinicians same side left',
        'all clinicians observer zone',
        'all clinicians active zone',
        'clinicians evenly split front back',
        'single clinician observer zone',
    ]

    testInterPersonDist(frames)
    testTableSummaries(frames,labels)
    testInterPersonSummaries(frames, labels)

    print("\nAll tests passed")

    return 

if __name__ == "__main__":
    testSceneGraph()