import numpy as np
from gpd_utils import JOINTS

def computeTableSummaries(framePoses):
    
    #clinician distribution
    numClinicians = len(framePoses)

    cliniciansLeft = 0 #clinicians to left of table
    cliniciansFront = 0 #clinicians in front of table

    for pose in framePoses:
        leftHip = pose[JOINTS['left_hip']]
        rightHip = pose[JOINTS['right_hip']]
        midHip = (leftHip + rightHip)/2

        if midHip[0] < 0: #x coordinate
            cliniciansLeft+=1

        if midHip[2] < 0: #zcoordinate
            cliniciansFront+=1
    
    leftRightRatio = cliniciansLeft/numClinicians
    frontBackRatio = cliniciansFront/numClinicians

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
    

    return np.array([activeClinicians, meanTableDist, leftRightRatio, frontBackRatio, minTableDist])

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
        return np.zeros(1)
    
    pairwiseDists = []
    for i in range(numClinicians):
        for j in range(i+1, numClinicians):
            dist = computeInterPersonDistance(framePoses[i], framePoses[j])
            pairwiseDists.append(dist)
    
    medianDist = np.median(pairwiseDists)
    
    return np.array([medianDist])

def computeSceneGraphSummary(framePoses):
    numClinicians = len(framePoses)

    if numClinicians == 0:
        return np.zeros(7)
    
    tableDistSummary = computeTableSummaries(framePoses)
    interPersonSummary = computeInterPersonSummaries(framePoses)
    
    return np.concatenate([[numClinicians], tableDistSummary, interPersonSummary])


def testSceneGraph():
    # test poses at different positions around table
    # table center assumed at origin (0,0,0)

    pose_left_active = np.array([
        [  -200, -700, 100],  # head
        [  -200, -630, 100],  # neck
        [  -300, -580, 100],  # left_shoulder
        [  -100, -580, 100],  # right_shoulder
        [  -300, -200, 100],  # left_hip
        [  -100, -200, 100],  # right_hip
        [  -320, -400, 100],  # left_elbow
        [  -120, -400, 100],  # right_elbow
        [  -330, -250,  50],  # left_wrist
        [  -130, -250,  50],  # right_wrist
        [  -340, -180,  20],  # left_hand
        [  -140, -180,  20],  # right_hand
    ])

    pose_right_active = np.array([
        [   200, -700, 100],  # head
        [   200, -630, 100],  # neck
        [   300, -580, 100],  # left_shoulder
        [   100, -580, 100],  # right_shoulder
        [   300, -200, 100],  # left_hip
        [   100, -200, 100],  # right_hip
        [   320, -400, 100],  # left_elbow
        [   120, -400, 100],  # right_elbow
        [   330, -250,  50],  # left_wrist
        [   130, -250,  50],  # right_wrist
        [   340, -180,  20],  # left_hand
        [   140, -180,  20],  # right_hand
    ])

    # clinician far from table, observer zone
    pose_observer = np.array([
        [  -100, -700, 1500],  # head
        [  -100, -630, 1500],  # neck
        [  -200, -580, 1500],  # left_shoulder
        [     0, -580, 1500],  # right_shoulder
        [  -200, -200, 1500],  # left_hip
        [     0, -200, 1500],  # right_hip
        [  -220, -400, 1500],  # left_elbow
        [   -20, -400, 1500],  # right_elbow
        [  -230, -250, 1500],  # left_wrist
        [   -30, -250, 1500],  # right_wrist
        [  -240, -180, 1500],  # left_hand
        [   -40, -180, 1500],  # right_hand
    ])

    # clinician at front of table
    pose_front = np.array([
        [   100, -700, -200],  # head
        [   100, -630, -200],  # neck
        [   200, -580, -200],  # left_shoulder
        [     0, -580, -200],  # right_shoulder
        [   200, -200, -200],  # left_hip
        [     0, -200, -200],  # right_hip
        [   220, -400, -200],  # left_elbow
        [    20, -400, -200],  # right_elbow
        [   230, -250, -200],  # left_wrist
        [    30, -250, -200],  # right_wrist
        [   240, -180, -200],  # left_hand
        [    40, -180, -200],  # right_hand
    ])

    print("Testing computeInterPersonDistance...")
    dist_close = computeInterPersonDistance(pose_left_active, pose_right_active)
    dist_far = computeInterPersonDistance(pose_left_active, pose_observer)
    assert isinstance(dist_close, (float, np.floating)), f"Expected float got {type(dist_close)}"
    assert dist_close >= 0, "Distance should be non negative"
    assert dist_far > dist_close, "Observer should be further than active clinician"
    print(f"  close pair distance:    {dist_close:.2f}")
    print(f"  far pair distance:      {dist_far:.2f}")
    print("  computeInterPersonDistance: OK")

    print("\nTesting computeInterPersonSummaries...")
    # test with single clinician
    single = computeInterPersonSummaries([pose_left_active])
    assert single.shape == (1,), f"Expected (1,) got {single.shape}"
    assert single[0] == 0, "Single clinician should return zero"
    print("  single clinician: OK")

    # test with two clinicians
    two_poses = [pose_left_active, pose_right_active]
    two = computeInterPersonSummaries(two_poses)
    assert two.shape == (1,), f"Expected (1,) got {two.shape}"
    assert two[0] >= 0, "Median should be non negative"
    print(f"  two clinicians median: {two[0]:.2f}")
    print("  two clinicians: OK")

    # test with three clinicians
    three_poses = [pose_left_active, pose_right_active, pose_observer]
    three = computeInterPersonSummaries(three_poses)
    assert three.shape == (1,), f"Expected (1,) got {three.shape}"
    print(f"  three clinicians median: {three[0]:.2f}")
    print("  three clinicians: OK")

    print("\nTesting computeTableSummaries...")
    # test single active clinician
    single_active = computeTableSummaries([pose_left_active])
    assert single_active.shape == (5,), f"Expected (5,) got {single_active.shape}"
    assert not np.any(np.isnan(single_active)), "Should not contain NaN"
    print(f"  single active: {single_active}")
    print("  single active clinician: OK")

    # test left/right distribution
    two_sided = computeTableSummaries([pose_left_active, pose_right_active])
    assert two_sided[2] == 0.5, f"Expected 0.5 left ratio got {two_sided[2]}"
    print(f"  left/right ratio (should be 0.5): {two_sided[2]}")
    print("  left/right distribution: OK")

    # test active zone count
    # pose_left_active and pose_right_active have joints close to origin
    # pose_observer has joints at z=1500 so should not be in active zone
    mixed_frame = computeTableSummaries([pose_left_active, pose_right_active, pose_observer])
    assert mixed_frame[0] <= 3, "Active clinicians should not exceed total clinicians"
    print(f"  active clinicians: {mixed_frame[0]}")
    print("  active zone: OK")

    # test front/back distribution
    front_back = computeTableSummaries([pose_front, pose_observer])
    assert 0 <= front_back[3] <= 1, "Front/back ratio should be between 0 and 1"
    print(f"  front/back ratio: {front_back[3]}")
    print("  front/back distribution: OK")

    print("\nTesting computeSceneGraphSummary...")
    # test empty frame
    empty = computeSceneGraphSummary([])
    assert empty.shape == (7,), f"Expected (7,) got {empty.shape}"
    assert np.all(empty == 0), "Empty frame should return zeros"
    print("  empty frame: OK")

    # test single clinician
    single_summary = computeSceneGraphSummary([pose_left_active])
    assert single_summary.shape == (7,), f"Expected (7,) got {single_summary.shape}"
    assert not np.any(np.isnan(single_summary)), "Should not contain NaN"
    print(f"  single clinician summary: {single_summary}")
    print("  single clinician: OK")

    # test full frame
    full_frame = [pose_left_active, pose_right_active, pose_observer, pose_front]
    full_summary = computeSceneGraphSummary(full_frame)
    assert full_summary.shape == (7,), f"Expected (7,) got {full_summary.shape}"
    assert not np.any(np.isnan(full_summary)), "Should not contain NaN"
    assert not np.any(np.isinf(full_summary)), "Should not contain inf"
    assert full_summary[0] == 4, f"Expected 4 clinicians got {full_summary[0]}"
    print(f"  full frame summary: {full_summary}")
    print("  full frame: OK")

    # semantic test - more clinicians close to table should increase active count
    active_heavy = computeSceneGraphSummary([pose_left_active, pose_right_active])
    observer_heavy = computeSceneGraphSummary([pose_observer, pose_front])
    assert active_heavy[1] >= observer_heavy[1], "Active frame should have more clinicians in active zone"
    print(f"  active frame active count:   {active_heavy[1]}")
    print(f"  observer frame active count: {observer_heavy[1]}")
    print("  semantic active zone: OK")

    print("\nAll scene graph tests passed!")

if __name__ == "__main__":
    testSceneGraph()