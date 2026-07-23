import numpy as np
import math as m
from general_utils import printKp, validateShape, printFrame, printOp 


#---Function and constants useful for building gpd vector-----

#mapping joints to indices in keypoints 
JOINTS = {'head': 0, 'neck': 1, 'left_shoulder': 2, 'right_shoulder': 3, 'left_hip': 4, 'right_hip': 5, 'left_elbow': 6, 'right_elbow': 7, 'left_wrist': 8, 'right_wrist': 9, 'left_hand': 10, 'right_hand': 11}

#defining planes
PLANES = ['torso', (2,6,10), (3,7,11)] #torso needs to be dervied 

#defining lines
LINES = [
    #12 direcrtly adjacent lines
    (0, 1),   # head - neck
    (1, 2),   # neck - left_shoulder
    (1, 3),   # neck - right_shoulder
    (2, 4),   # left_shoulder - left_hip
    (3, 5),   # right_shoulder - right_hip
    (4, 5),   # left_hip - right_hip
    (2, 6),   # left_shoulder - left_elbow
    (3, 7),   # right_shoulder - right_elbow
    (6, 8),   # left_elbow - left_wrist
    (7, 9),   # right_elbow - right_wrist
    (8, 10),  # left_wrist - left_hand
    (9, 11),  # right_wrist - right_hand

    #Type 2 lines
    (0, 2),   # head - left_shoulder
    (0, 3),   # head - right_shoulder
    (10, 6),  # left_hand - left_shoulder
    (11, 7),  # right_hand - right_shoulder

    #Type 3 lines
    (0, 10),  # head - left_hand
    (0, 11),  # head - right_hand
    (10, 11), # left_hand - right_hand
]


#function to normalize coords WRT to hip center 
def normalizePose(keypoints):
    #compute mid hip
    leftHip = keypoints[JOINTS['left_hip']]
    rightHip = keypoints[JOINTS['right_hip']]
    midHip = (leftHip + rightHip)/2

    #Joints wrt hip = joint coord - hip coord
    normalizedCoords = keypoints - midHip
    
    return normalizedCoords


#derives chest keypoint
def getChest(keypoints):
    leftShoulder = keypoints[JOINTS['left_shoulder']]
    rightShoulder = keypoints[JOINTS['right_shoulder']]
    return (leftShoulder + rightShoulder) / 2

#computes unit vector
def unit(vector):
    norm = np.linalg.norm(vector) 

    #preventing zero division error
    if norm > 0:
        unit = vector/norm
            
    else:
        unit = np.zeros(3) 
    
    return unit


#helper to find index in flat pairwise distance array
def getPairIndex(i, j, n=12): #works given i < j 
    return i * (n-1) - (i*(i-1))//2 + (j-i-1)

#helper for joint line distances: computes area given side lengths
def heronFormula(pairwiseDist, j, j1, j2):

    j_j1 = getPairIndex(min(j,j1), max(j,j1))
    j_j2 = getPairIndex(min(j,j2), max(j,j2))
    j1_j2 = getPairIndex(min(j1,j2), max(j1,j2))

    a = pairwiseDist[j_j1]
    b = pairwiseDist[j_j2]
    c = pairwiseDist[j1_j2]

    s = (a+b+c)/2
    product = s*(s-a)*(s-b)*(s-c)
    A = m.sqrt(max(0.0, product)) #ensuring no negative inputs
    return A

#-----8 static features---
#joint xyz coordinates
def computeJc(keypoints):
    return keypoints.flatten() #turning 12x3 into 36 values

#joint-joint distances
def computeJjd(keypoints):

    jjd = []
    numJoints = len(keypoints)
    for i in range(numJoints):
        for j in range(i+1, numJoints):
            dist = np.linalg.norm(keypoints[i] - keypoints[j])
            jjd.append(dist)
    
    return np.array(jjd)

#joint-joint orientations        
def computeJjo(keypoints):
    
    jjo = []
    numJoints = len(keypoints)

    for i in range(numJoints):
        for j in range(i+1, numJoints):
            vector =  keypoints[j] - keypoints[i] 
            jjo.append(unit(vector))

    jjo = np.array(jjo)   
    
    return jjo, jjo.flatten() #unflatened to facilitate future computations, flattened version to add to feature vector


#joint-line distances
def computeJld(keypoints, pairwiseDist):

    jld = []
    for i in range(len(LINES)):
        endpoints = LINES[i]
        for j in range(len(keypoints)):
            if j == endpoints[0] or j==endpoints[1]:
                continue
            
            distNum = 2*heronFormula(pairwiseDist, j, endpoints[0], endpoints[1]) #numerator of equation
            distDen = pairwiseDist[getPairIndex(min(endpoints[0], endpoints[1]), max(endpoints[0], endpoints[1]))] #denominator of equation
            dist = distNum /distDen
            jld.append(dist)

    return np.array(jld)

#line-line angles     
def computeLla(jjo):
    lla = []
    numLines = len(LINES)

    for i in range(numLines):
        for j in range(i+1, numLines):
            l1 = LINES[i] #j1j2
            l2 = LINES[j] #j1'j2'

            #finding idx in joint orientations
            l1_idx = getPairIndex(min(l1[0], l1[1]), max(l1[0], l1[1]))
            l2_idx = getPairIndex(min(l2[0], l2[1]), max(l2[0], l2[1]))

            #1x3 array 
            l1_o = jjo[l1_idx] 
            l2_o = jjo[l2_idx]

            dotProd = np.clip(np.dot(l1_o, l2_o), -1.0 , 1.0) #.clip keeps dot product within bounds of arccos
            angle = np.arccos(dotProd)
            lla.append(angle)

    return np.array(lla)

#joint-plane distances
def computeJpd(keypoints, jjo):
    jpd = []

    #derive chest
    chest = getChest(keypoints)


    for i in range(len(PLANES)):
        points = PLANES[i]

        if points == 'torso':
            points = (-1, 1, 0) #-1 flag indicating its chest kp


        for j in range(len(keypoints)):
            if j in points:
                continue

            #dealing for torso plane
            if -1 in points:
                o1 = unit(keypoints[j]- chest) #jjo j1,j
                o2 = unit(keypoints[points[1]]- chest) #jjo j1,j2
                o3 = unit(keypoints[points[2]]- chest)#jjo j1,j3

            else:
                o1_idx = getPairIndex(min(points[0], j), max(points[0], j))
                o2_idx = getPairIndex(min(points[0], points[1]), max(points[0], points[1]))
                o3_idx = getPairIndex(min(points[0], points[2]), max(points[0], points[2]))

                #ternary operator ensures  vector is in right direction

                o1 = jjo[o1_idx] if points[0] < j else -jjo[o1_idx] #jjo j1,j
                o2 = jjo[o2_idx] if points[0] < points[1] else -jjo[o2_idx] #jjo j1,j2
                o3 = jjo[o3_idx] if points[0] < points[2] else -jjo[o3_idx] #jjo j1,j3
                


            cross = np.cross(o2,o3) #cross product
            dist = np.dot(o1,unit(cross))

            jpd.append(dist)

    return np.array(jpd)
         
#line plane angles
def computeLpa(keypoints, jjo):
    lpa = []

    #derive chest
    chest = getChest(keypoints)

    for i in range(len(PLANES)):
        points = PLANES[i]

        if points == 'torso':
            points = (-1, 1, 0) #-1 flag indicating its chest kp

        for j in range(len(LINES)):
            endpoints = LINES[j]

            o1_idx = getPairIndex(min(endpoints[0], endpoints[1]), max(endpoints[0], endpoints[1]))
            o1 = jjo[o1_idx] if endpoints[0] < endpoints[1] else -jjo[o1_idx] #jjo j1,j2

            

            #dealing for torso plane
            if -1 in points:
                o2 = unit(keypoints[points[1]]- chest) #jjo j1',j2'
                o3 = unit(keypoints[points[2]]- chest) #jjo j1',j3'

            else:
                o2_idx = getPairIndex(min(points[0], points[1]), max(points[0], points[1]))
                o3_idx = getPairIndex(min(points[0], points[2]), max(points[0], points[2]))


                o2 = jjo[o2_idx] if points[0] < points[1] else -jjo[o2_idx] #jjo j1',j2'
                o3 = jjo[o3_idx] if points[0] < points[2] else -jjo[o3_idx] #jjo j1',j3'

            
            cross = np.cross(o2,o3)
            temp = np.dot(o1, unit(cross)) #intermediate step
            temp = np.clip(temp, -1.0, 1.0) #.clip keeps value within bounds of arccos
            angle = np.arccos(temp)

            lpa.append(angle)


    return np.array(lpa)

#plane-plane angles
def computePpa(keypoints, jjo):
    ppa = []

    chest = getChest(keypoints)

    for i in range(len(PLANES)):
        p1 = PLANES[i] #j1->j2->j3

        if p1 == 'torso':
            p1 = (-1,1,0)
            o1 = unit(keypoints[p1[1]]- chest) #jjo j1,j2
            o2 = unit(keypoints[p1[2]]- chest) #jjo j1,j3

        else: 
            o1_idx = getPairIndex(min(p1[0], p1[1]), max(p1[0], p1[1]))
            o2_idx = getPairIndex(min(p1[0], p1[2]), max(p1[0], p1[2]))

            o1 = jjo[o1_idx] 
            o2 = jjo[o2_idx]

        cross1 = unit(np.cross(o1,o2))  

        for j in range(i+1, len(PLANES)):
            p2 = PLANES[j] #j1'->j2'->j3'

            o3_idx = getPairIndex(min(p2[0], p2[1]), max(p2[0], p2[1]))
            o4_idx = getPairIndex(min(p2[0], p2[2]), max(p2[0], p2[2]))

            o3 = jjo[o3_idx] 
            o4 = jjo[o4_idx]

            
            cross2 = unit(np.cross(o3,o4))
            dot = np.clip(np.dot(cross1,cross2), -1.0, 1.0) #.clip keeps value within bounds of arccos
            angle = np.arccos(dot)

            ppa.append(angle)

    return np.array(ppa)         

#computes full gpd vector 
def computeGpd(keypoints):
    #normalizing coords wrt hip
    normalizedKp = normalizePose(keypoints)

    #computing static features
    jc = computeJc(normalizedKp)
    jjd = computeJjd(normalizedKp)
    jjo, jjoFlat = computeJjo(normalizedKp)
    jld = computeJld(normalizedKp, jjd)
    lla = computeLla(jjo)
    jpd = computeJpd(normalizedKp, jjo)
    lpa = computeLpa(normalizedKp, jjo)
    ppa = computePpa(normalizedKp, jjo)

    #concatenation into full feature vector
    gpd = np.concatenate([jc, jjd, jjoFlat, jld, lla, jpd, lpa, ppa])

    return gpd 

#---unit testing-------
def testNormalizePose():
    print("Testing normalizePose...")

    # test 1 - basic shape check
    pose = np.array([
        [630.65, -498.06, 2394.90],  # head
        [650.60, -435.67, 2467.15],  # neck
        [765.23, -415.12, 2567.27],  # left_shoulder
        [471.51, -417.43, 2576.98],  # right_shoulder
        [762.86, -103.45, 2816.92],  # left_hip
        [493.88,  -88.01, 2806.61],  # right_hip
        [739.04, -224.36, 2560.00],  # left_elbow
        [334.11, -217.05, 2561.78],  # right_elbow
        [737.65, -122.52, 2450.00],  # left_wrist
        [478.40, -167.64, 2436.25],  # right_wrist
        [787.65,  -72.52, 2500.00],  # left_hand
        [541.82, -182.77, 2388.57],  # right_hand
    ])

    #testing that shape is retained 
    normalized = normalizePose(pose)
    assert normalized.shape == (12,3), f"Expected (12,3), got {normalized.shape}"
    print( "    test 1 - shape preservation: OK")

    #hip midpoint must be the origin
    hip_mid = (normalized[JOINTS['left_hip']] + normalized[JOINTS['right_hip']]) / 2
    assert np.allclose(hip_mid, 0), f"Hip midpoint should be at origin got {hip_mid}"
    print("  test 2 - hip midpoint at origin: OK")

    #testing that relative distances are preserved
    for i in range(12):
        for j in range(i+1, 12):
            dist_before = np.linalg.norm(pose[i] - pose[j])
            dist_after = np.linalg.norm(normalized[i] - normalized[j])
            assert np.isclose(dist_before, dist_after), f"Distance between joints {i} and {j} changed after normalization"
    print("  test 3 - relative distances preserved: OK")

    #alr centered pose should remain the same
    centered_pose = pose.copy()
    hip_mid = (centered_pose[JOINTS['left_hip']] + centered_pose[JOINTS['right_hip']]) / 2
    centered_pose = centered_pose - hip_mid
    renormalized = normalizePose(centered_pose)
    assert np.allclose(renormalized, centered_pose), "Already centered pose should be unchanged"
    print("  test 4 - idempotent on already centered pose: OK")

    # shifting all joints by same amount should give same normalized result
    shift = np.array([100, 200, 300])
    shifted_pose = pose + shift
    normalized_original = normalizePose(pose)
    normalized_shifted = normalizePose(shifted_pose)
    assert np.allclose(normalized_original, normalized_shifted), "Normalization should be translation invariant"
    print("  test 5 - translation invariance: OK")

    # if left and right hips are symmetric around x=0, hip mid x should be 0
    symmetric_pose = pose.copy()
    symmetric_pose[JOINTS['left_hip']][0] = -200
    symmetric_pose[JOINTS['right_hip']][0] = 200
    normalized_sym = normalizePose(symmetric_pose)
    assert np.isclose(normalized_sym[JOINTS['left_hip']][0], -normalized_sym[JOINTS['right_hip']][0]), "Symmetric hips should remain symmetric after normalization"
    print("  test 6 - symmetry preserved: OK")

    #large coordinate values (edge case)
    large_pose = pose * 1e6
    normalized_large = normalizePose(large_pose)
    assert not np.any(np.isnan(normalized_large)), "Large coordinates should not produce NaN"
    assert not np.any(np.isinf(normalized_large)), "Large coordinates should not produce inf"
    print("  test 7 - large coordinate values: OK")

    print("\nAll normalizePose tests passed!")
    
def testGetChest():
    print("Testing getChest...")

    pose = np.array([
        [630.65, -498.06, 2394.90],  # head
        [650.60, -435.67, 2467.15],  # neck
        [765.23, -415.12, 2567.27],  # left_shoulder
        [471.51, -417.43, 2576.98],  # right_shoulder
        [762.86, -103.45, 2816.92],  # left_hip
        [493.88,  -88.01, 2806.61],  # right_hip
        [739.04, -224.36, 2560.00],  # left_elbow
        [334.11, -217.05, 2561.78],  # right_elbow
        [737.65, -122.52, 2450.00],  # left_wrist
        [478.40, -167.64, 2436.25],  # right_wrist
        [787.65,  -72.52, 2500.00],  # left_hand
        [541.82, -182.77, 2388.57],  # right_hand
    ])

    pose = normalizePose(pose)
   
    #shape
    chest = getChest(pose) 
    assert chest.shape == (3,), f"Expected (3,) got {chest}"
    print("  test 1 - output shape: OK")


    #large coordinate values
    large_pose = pose * 1e6
    chest_large = getChest(large_pose)
    assert not np.any(np.isnan(chest_large)), "Large coordinates should not produce NaN"
    assert not np.any(np.isinf(chest_large)), "Large coordinates should not produce inf"
    print("  test 2 - large coordinate values: OK")

    #chest is midpoint of shoulders
    left_shoulder = pose[JOINTS['left_shoulder']]
    right_shoulder = pose[JOINTS['right_shoulder']]
    expected_chest = (left_shoulder + right_shoulder) / 2
    assert np.allclose(chest, expected_chest), f"Chest should be midpoint of shoulders got {chest}"
    print("  test 2 - correct midpoint: OK") 


    #translation invariance
    shift = np.array([100, 200, 300])
    shifted_pose = pose + shift
    chest_shifted = getChest(shifted_pose)
    assert np.allclose(chest_shifted, chest + shift), "Chest should shift by same amount as pose"
    print("  test 4 - translation invariance: OK")

    print("\nAll getChest tests passed!")

def testUnit():
    print("Testing unit...")

    #basic unit vector 
    v = np.array([3.0, 0.0, 0.0])
    u = unit(v)
    assert u.shape == (3,), f"Expected (3,) got {u.shape}"
    assert np.isclose(np.linalg.norm(u), 1.0), f"Unit vector should have norm 1 got {np.linalg.norm(u)}"
    assert np.allclose(u, [1.0, 0.0, 0.0]), f"Expected [1,0,0] got {u}"
    print("  test 1 - basic unit vector: OK")

    #preservation of direction
    v3 = np.array([1.0, 1.0, 1.0])
    u3 = unit(v3)
    assert np.isclose(np.linalg.norm(u3), 1.0), "Norm should be 1"
    assert np.allclose(u3, v3 / np.linalg.norm(v3)), "Direction should be preserved"
    print("  test 2 - direction preserved: OK")

    #large values
    v5 = np.array([1e6, 1e6, 1e6])
    u5 = unit(v5)
    assert np.isclose(np.linalg.norm(u5), 1.0), "Large vector should normalize correctly"
    assert not np.any(np.isnan(u5)), "Large vector should not produce NaN"
    print("  test 3 - large vector: OK")

    #small values
    v6 = np.array([1e-10, 1e-10, 1e-10])
    u6 = unit(v6)
    assert np.isclose(np.linalg.norm(u6), 1.0), "Small vector should normalize correctly"
    assert not np.any(np.isnan(u6)), "Small vector should not produce NaN"
    print("  test 4 - small vector: OK")

    #zero vector
    v7 = np.array([0.0, 0.0, 0.0])
    u7 = unit(v7)
    assert not np.any(np.isnan(u7)), "Zero vector should not produce NaN"
    assert not np.any(np.isinf(u7)), "Zero vector should not produce inf"
    assert np.allclose(u7, [0.0, 0.0, 0.0]), "Zero vector should return zero vector"
    print("  test 5 - zero vector: OK")


    print("\nAll unit tests passed!")

def testGetPairIdx():
    print("Testing getPairIndex...")

    # test 1 - first pair should be index 0
    idx = getPairIndex(0, 1)
    assert idx == 0, f"Expected 0 got {idx}"
    print("  test 1 - first pair (0,1) = index 0: OK")

    # test 2 - second pair should be index 1
    idx = getPairIndex(0, 2)
    assert idx == 1, f"Expected 1 got {idx}"
    print("  test 2 - second pair (0,2) = index 1: OK")

    # test 3 - last pair of first row
    idx = getPairIndex(0, 11)
    assert idx == 10, f"Expected 10 got {idx}"
    print("  test 3 - last pair of first row (0,11) = index 10: OK")

    # test 4 - first pair of second row
    idx = getPairIndex(1, 2)
    assert idx == 11, f"Expected 11 got {idx}"
    print("  test 4 - first pair of second row (1,2) = index 11: OK")

    # test 5 - last possible pair
    # last pair is (10, 11) which should be index 65 (C(12,2)-1)
    idx = getPairIndex(10, 11)
    assert idx == 65, f"Expected 65 got {idx}"
    print("  test 5 - last pair (10,11) = index 65: OK")

    # test 6 - total unique pairs should be C(12,2) = 66
    indices = []
    for i in range(12):
        for j in range(i+1, 12):
            indices.append(getPairIndex(i, j))
    assert len(indices) == 66, f"Expected 66 pairs got {len(indices)}"
    print("  test 6 - total pairs count = 66: OK")

    # test 7 - all indices should be unique
    assert len(set(indices)) == 66, "All pair indices should be unique"
    print("  test 7 - all indices unique: OK")

    # test 8 - all indices should be in valid range
    assert min(indices) == 0, f"Minimum index should be 0 got {min(indices)}"
    assert max(indices) == 65, f"Maximum index should be 65 got {max(indices)}"
    print("  test 8 - indices in valid range [0,65]: OK")

    print("\nAll getPairIndex tests passed!")

def testJc(normalized):
    jc = computeJc(normalized)
    assert jc.shape == (36,), f"Expected (36,) got {jc.shape}"
    print("  computeJc: OK")

def testJjd(normalized):
    jjd = computeJjd(normalized)
    assert jjd.shape == (66,), f"Expected (66,) got {jjd.shape}"
    assert np.all(jjd >= 0), "All distances should be non negative"
    assert not np.any(np.isnan(jjd)), "Should not contain NaN"
    assert not np.any(np.isinf(jjd)), "Should not contain inf"
    print("  computeJjd: OK")

def testJjo(normalized):
    jjo, jjoFlat = computeJjo(normalized)
    
    #shape
    assert jjo.shape == (66, 3), f"Expected (66,3) got {jjo.shape}"
    assert jjoFlat.shape == (198,), f"Expected (198,) got {jjoFlat.shape}"
    
    #checking that vectors are unit length
    norms = np.linalg.norm(jjo, axis=1)
    assert np.allclose(norms, 1.0), "All orientation vectors should be unit length"
    
    #verifying against know orientation
    head = normalized[JOINTS['head']]
    neck = normalized[JOINTS['neck']]
    expected_orientation = unit(neck-head)
    idx = getPairIndex(min(JOINTS['neck'], JOINTS['head']), max(JOINTS['neck'], JOINTS['head']))

    assert np.allclose(jjo[idx], expected_orientation), f"Expected {expected_orientation} got {jjo[idx]}"
    
    #both jjo arrays contain the same values
    assert np.allclose(jjoFlat, jjo.flatten()), "Flat output should be reshape of matrix"

    #no NaN of inf
    assert not np.any(np.isnan(jjo)), "Matrix should not contain NaN"
    assert not np.any(np.isinf(jjo)), "Matrix should not contain inf"
    assert not np.any(np.isnan(jjoFlat)), "Flat should not contain NaN"
    assert not np.any(np.isinf(jjoFlat)), "Flat should not contain inf"

    print("  computeJjo: OK")   
    
def testJld(normalized, jjd):
    jld = computeJld(normalized, jjd)

    #shape
    assert jld.shape == (190,), f"Expected (190,) got {jld.shape}"
    
    #non negative distances
    assert np.all(jld >= 0), "All distances should be non negative"

    #no NaN or inf
    assert not np.any(np.isnan(jld)), "Should not contain NaN"
    assert not np.any(np.isinf(jld)), "Should not contain inf"

    print("  computeJld: OK")

def testLla(normalized, jjo):
    lla = computeLla(jjo)

    #basic shape 
    assert lla.shape == (171,), f"Expected (171,) got {lla.shape}"
    assert np.all(lla >= 0) and np.all(lla <= np.pi), "All angles should be between 0 and pi"

    # no NaN or inf
    assert not np.any(np.isnan(lla)), "Should not contain NaN"
    assert not np.any(np.isinf(lla)), "Should not contain inf"


    print("  computeLla: OK")

def testJpd(normalized, jjo):    
    jpd = computeJpd(normalized, jjo)
    
    #shape
    assert jpd.shape == (28,), f"Expected (28,) got {jpd.shape}"
    print("  computeJpd: OK")

    #no NaN or inf
    assert not np.any(np.isnan(jpd)), "Should not contain NaN"
    assert not np.any(np.isinf(jpd)), "Should not contain inf"

    #known distance verification for arm plane
    #place left arm joints in xy plane and verify distance of known joint
    known_pose = normalized.copy()
    known_pose[JOINTS['left_shoulder']] = np.array([0.0,  0.0, 0.0])
    known_pose[JOINTS['left_elbow']]    = np.array([10.0, 0.0, 0.0])
    known_pose[JOINTS['left_hand']]     = np.array([0.0, 10.0, 0.0])
    # place neck 5 units above the left arm plane (z direction)
    known_pose[JOINTS['neck']]          = np.array([0.0,  0.0, 5.0])
    
    known_jjo, _ = computeJjo(known_pose)
    known_jpd = computeJpd(known_pose, known_jjo)


    # neck distance to left arm plane should be 1
    # left arm plane is PLANES[1] = (2,6,10), neck is not a vertex
    # find neck's position in output for left arm plane
    plane_idx = 1  # left arm plane
    valid_joints_before_neck = sum(1 for j in range(JOINTS['neck'])
                                    if j not in PLANES[plane_idx])
    feature_idx = 10 + valid_joints_before_neck  # 9 joints for torso plane come first

    assert np.isclose(abs(known_jpd[feature_idx]), 1.0, atol=1e-6), f"Expected distance 1.0 got {abs(known_jpd[feature_idx]):.6f}"



    #joint on plane should give zero distance (look into dropping)
    on_plane_pose = normalized.copy()
    on_plane_pose[JOINTS['left_shoulder']] = np.array([0.0,  0.0, 0.0])
    on_plane_pose[JOINTS['left_elbow']]    = np.array([10.0, 0.0, 0.0])
    on_plane_pose[JOINTS['left_hand']]     = np.array([0.0, 10.0, 0.0])
    # place neck exactly on the left arm plane
    on_plane_pose[JOINTS['neck']]          = np.array([5.0, 5.0, 0.0])
    on_plane_jjo, _ = computeJjo(on_plane_pose)
    on_plane_jpd = computeJpd(on_plane_pose, on_plane_jjo)
    assert np.isclose(on_plane_jpd[feature_idx], 0.0, atol=1e-6), \
        f"Joint on plane should give zero distance got {on_plane_jpd[feature_idx]:.6f}"


    #torso plane handled correctly
    #verify torso plane features are computed without error
    # torso plane uses derived chest point
    torso_pose = normalized.copy()
    torso_pose[JOINTS['left_shoulder']]  = np.array([-10.0, 0.0, 0.0])
    torso_pose[JOINTS['right_shoulder']] = np.array([ 10.0, 0.0, 0.0])
    torso_pose[JOINTS['neck']]           = np.array([  0.0, 10.0, 0.0])
    torso_pose[JOINTS['head']]           = np.array([  0.0, 20.0, 0.0])
    # chest = midpoint of shoulders = [0, 0, 0]
    # torso plane spanned by chest[0,0,0], neck[0,10,0], head[0,20,0]
    torso_jjo, _ = computeJjo(torso_pose)
    torso_jpd = computeJpd(torso_pose, torso_jjo)
    assert not np.any(np.isnan(torso_jpd)), "Torso plane should not produce NaN"
    assert not np.any(np.isinf(torso_jpd)), "Torso plane should not produce inf"

def testLpa(normalized, jjo):
    lpa = computeLpa(normalized, jjo)

    #basic shape 
    assert lpa.shape == (57,), f"Expected (57,) got {lpa.shape}"
    assert np.all(lpa >= 0) and np.all(lpa <= np.pi), "All angles should be between 0 and pi"

    #no NaN or inf
    assert not np.any(np.isnan(lpa)), "Should not contain NaN"
    assert not np.any(np.isinf(lpa)), "Should not contain inf"

    print("  computeLpa: OK")

    # known angle: line perpendicular to plane normal should give 0
    # place left arm plane in xy plane (z=0)
    # place a line along z axis - perpendicular to plane
    # angle between line and plane normal (z axis) should be 0
    # angle between line and plane itself should be pi/2
    known_pose = normalized.copy()
    known_pose[JOINTS['left_shoulder']] = np.array([0.0,  0.0, 0.0])
    known_pose[JOINTS['left_elbow']]    = np.array([10.0, 0.0, 0.0])
    known_pose[JOINTS['left_hand']]     = np.array([0.0, 10.0, 0.0])
    # place neck-head line along z axis
    known_pose[JOINTS['neck']]          = np.array([0.0, 0.0,  10.0])
    known_pose[JOINTS['head']]          = np.array([0.0, 0.0, 0.0])
    known_jjo, _ = computeJjo(known_pose)
    known_lpa = computeLpa(known_pose, known_jjo)
    # neck-head line vs left arm plane
    # normal of left arm plane is z axis
    # angle between neck-head (z axis) and normal (z axis) should be 0
    line_idx = LINES.index((JOINTS['neck'], JOINTS['head']))\
        if (JOINTS['neck'], JOINTS['head']) in LINES \
        else LINES.index((min(JOINTS['neck'], JOINTS['head']),
                        max(JOINTS['neck'], JOINTS['head'])))
    plane_idx = 1  # left arm plane
    feature_idx = plane_idx * 19 + line_idx

 
    assert np.isclose(known_lpa[feature_idx], 0.0, atol=1e-6), f"Line along normal should give angle 0 got {known_lpa[feature_idx]:.6f}"

    # line parallel to plane should give pi/2
    # place neck-head line in xy plane parallel to left arm plane
    known_pose2 = normalized.copy()
    known_pose2[JOINTS['left_shoulder']] = np.array([0.0,  0.0, 0.0])
    known_pose2[JOINTS['left_elbow']]    = np.array([10.0, 0.0, 0.0])
    known_pose2[JOINTS['left_hand']]     = np.array([0.0, 10.0, 0.0])
    known_pose2[JOINTS['neck']]          = np.array([0.0, 0.0, 0.0])
    known_pose2[JOINTS['head']]          = np.array([10.0, 0.0, 0.0])
    known_jjo2, _ = computeJjo(known_pose2)
    known_lpa2 = computeLpa(known_pose2, known_jjo2)
    assert np.isclose(known_lpa2[feature_idx], np.pi/2, atol=1e-6), f"Line parallel to plane should give pi/2 got {known_lpa2[feature_idx]:.6f}"

    # test 7 - torso plane handled correctly
    torso_pose = normalized.copy()
    torso_pose[JOINTS['left_shoulder']]  = np.array([-10.0,  0.0, 0.0])
    torso_pose[JOINTS['right_shoulder']] = np.array([ 10.0,  0.0, 0.0])
    torso_pose[JOINTS['neck']]           = np.array([  0.0, 10.0, 0.0])
    torso_pose[JOINTS['head']]           = np.array([  0.0, 20.0, 0.0])
    torso_jjo, _ = computeJjo(torso_pose)
    torso_lpa = computeLpa(torso_pose, torso_jjo)
    assert not np.any(np.isnan(torso_lpa)), "Torso plane should not produce NaN"
    assert not np.any(np.isinf(torso_lpa)), "Torso plane should not produce inf"
    assert np.all(torso_lpa >= 0) and np.all(torso_lpa <= np.pi), "Torso plane angles should be in valid range"

def testPpa(normalized, jjo):
    ppa = computePpa(normalized, jjo)

    #basic shape
    assert ppa.shape == (3,), f"Expected (3,) got {ppa.shape}"
    assert np.all(ppa >= 0) and np.all(ppa <= np.pi), "All angles should be between 0 and pi"

    #no NaN or inf
    assert not np.any(np.isnan(ppa)), "Should not contain NaN"
    assert not np.any(np.isinf(ppa)), "Should not contain inf"

    # parallel planes should give angle 0
    # place left and right arm planes both in xy plane
    parallel_pose = normalized.copy()
    parallel_pose[JOINTS['left_shoulder']]  = np.array([ 0.0,  0.0, 0.0])
    parallel_pose[JOINTS['left_elbow']]     = np.array([10.0,  0.0, 0.0])
    parallel_pose[JOINTS['left_hand']]      = np.array([ 0.0, 10.0, 0.0])
    parallel_pose[JOINTS['right_shoulder']] = np.array([ 5.0,  0.0, 0.0])
    parallel_pose[JOINTS['right_elbow']]    = np.array([15.0,  0.0, 0.0])
    parallel_pose[JOINTS['right_hand']]     = np.array([ 5.0, 10.0, 0.0])
    parallel_jjo, _ = computeJjo(parallel_pose)
    parallel_ppa = computePpa(parallel_pose, parallel_jjo)
    # left arm vs right arm is index 0 (first pair in outer loop i=1, j=2)
    assert np.isclose(parallel_ppa[2], 0.0, atol=1e-6), f"Parallel planes should give angle 0 got {parallel_ppa[0]:.6f}"


    # perpendicular planes should give angle pi/2
    perp_pose = normalized.copy()
    perp_pose[JOINTS['left_shoulder']]  = np.array([ 0.0,  0.0, 0.0])
    perp_pose[JOINTS['left_elbow']]     = np.array([10.0,  0.0, 0.0])
    perp_pose[JOINTS['left_hand']]      = np.array([ 0.0, 10.0, 0.0])  # left arm in xy plane
    perp_pose[JOINTS['right_shoulder']] = np.array([ 0.0,  0.0, 0.0])
    perp_pose[JOINTS['right_elbow']]    = np.array([10.0,  0.0, 0.0])
    perp_pose[JOINTS['right_hand']]     = np.array([ 0.0,  0.0, 10.0])  # right arm in xz plane
    perp_jjo, _ = computeJjo(perp_pose)
    perp_ppa = computePpa(perp_pose, perp_jjo)
    assert np.isclose(perp_ppa[2], np.pi/2, atol=1e-6), f"Perpendicular planes should give pi/2 got {perp_ppa[0]:.6f}"

    # test 7 - torso plane handled correctly
    # torso plane is PLANES[0] so it appears in pairs (0,1) and (0,2)
    torso_pose = normalized.copy()
    torso_pose[JOINTS['left_shoulder']]  = np.array([-10.0,  0.0, 0.0])
    torso_pose[JOINTS['right_shoulder']] = np.array([ 10.0,  0.0, 0.0])
    torso_pose[JOINTS['neck']]           = np.array([  0.0, 10.0, 0.0])
    torso_pose[JOINTS['head']]           = np.array([  0.0, 20.0, 0.0])
    torso_pose[JOINTS['left_elbow']]     = np.array([-10.0,  0.0, 0.0])
    torso_pose[JOINTS['left_hand']]      = np.array([-10.0, 10.0, 0.0])
    torso_pose[JOINTS['right_elbow']]    = np.array([ 10.0,  0.0, 0.0])
    torso_pose[JOINTS['right_hand']]     = np.array([ 10.0, 10.0, 0.0])
    torso_jjo, _ = computeJjo(torso_pose)
    torso_ppa = computePpa(torso_pose, torso_jjo)
    assert not np.any(np.isnan(torso_ppa)), "Torso plane should not produce NaN"
    assert not np.any(np.isinf(torso_ppa)), "Torso plane should not produce inf"
    assert np.all(torso_ppa >= 0) and np.all(torso_ppa <= np.pi), "Torso plane angles should be in valid range"
    print("  computePpa: OK")
    
#Test suite
def testGpd():
    # AI generated sample keypoints  (12x3 numpy array)
    test_kp_1 = np.array([
        [630.65, -498.06, 2394.90],  # head
        [650.60, -435.67, 2467.15],  # neck
        [765.23, -415.12, 2567.27],  # left_shoulder
        [471.51, -417.43, 2576.98],  # right_shoulder
        [762.86, -103.45, 2816.92],  # left_hip
        [493.88, -88.01,  2806.61],  # right_hip
        [739.04, -224.36, 2560.00],  # left_elbow
        [334.11, -217.05, 2561.78],  # right_elbow
        [737.65, -122.52, 2450.00],  # left_wrist
        [478.40, -167.64, 2436.25],  # right_wrist
        [787.65, -72.52,  2500.00],  # left_hand
        [541.82, -182.77, 2388.57],  # right_hand
    ])

    # clinician leaning over table
    test_kp_2 = np.array([
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

    # clinician standing upright arms at sides
    test_kp_3 = np.array([
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

    # clinician reaching across to one side
    test_kp_4 = np.array([
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

    test_keypoints = [test_kp_1, test_kp_2, test_kp_3, test_kp_4]

    print("Testing helpers...")

    testNormalizePose()
    testGetChest()
    testUnit()
    testGetPairIdx()


    for i, test_kp in enumerate(test_keypoints, start = 1):
        print(f"\nTesting sample {i}")

        
        normalized = normalizePose(test_kp)
        assert normalized.shape == (12, 3), f"Expected (12,3) got {normalized.shape}"
        print("  normalizedPose: OK")

        testJc(normalized)
        testJjd(normalized)
        testJjo(normalized)

        jjd = computeJjd(normalized)

        testJld(normalized,jjd)

        jjo, jjoFlat = computeJjo(normalized)

        testLla(normalized, jjo)
        testJpd(normalized, jjo)
        testLpa(normalized,jjo)
        testPpa(normalized, jjo)

        gpd = computeGpd(test_kp)
        assert gpd.shape == (749,), f"Expected (749,) got {gpd.shape}"
        assert not np.any(np.isnan(gpd)), "GPD vector should not contain NaN values"
        assert not np.any(np.isinf(gpd)), "GPD vector should not contain inf values"
        print("  computeGpd: OK")

        print("All tests passed")

    return 


if __name__ == "__main__":
    testGpd() #all tests pass




