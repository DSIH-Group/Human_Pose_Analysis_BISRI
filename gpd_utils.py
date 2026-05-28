import pickle 
import numpy as np
import math as m
from general_utils import printKp, validateShape, printFrame, printOp 


try:
    with open('cleanedAnnots.pkl', 'rb') as f:
        annots = pickle.load(f)
except:
    print("Error")

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
            vector = keypoints[j] - keypoints[i]
            jjo.append(unit(vector))

    jjo = np.array(jjo)   
    
    return jjo, jjo.flatten() #unflatened to facilitate future computations, flattened version to add to feature vector


#helper to find index in flat pairwise distance array
def getPairIndex(i, j, n=12): #works given i < j 
    return i * (n-1) - (i*(i-1))//2 + (j-i-1)

#helper for jld 
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

                o1 = jjo[o1_idx] #jjo j1,j
                o2 = jjo[o2_idx] #jjo j1,j2
                o3 = jjo[o3_idx] #jjo j1,j3

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
            o1 = jjo[o1_idx] #jjo j1,j2
            

            #dealing for torso plane
            if -1 in points:
                o2 = unit(keypoints[points[1]]- chest) #jjo j1',j2'
                o3 = unit(keypoints[points[2]]- chest) #jjo j1',j3'

            else:
                o2_idx = getPairIndex(min(points[0], points[1]), max(points[0], points[1]))
                o3_idx = getPairIndex(min(points[0], points[2]), max(points[0], points[2]))

                o2 = jjo[o2_idx] #jjo j1',j2'
                o3 = jjo[o3_idx] #jjo j1',j3'
            
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

#quick test ensuring main functionality works 
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

    for i, test_kp in enumerate(test_keypoints, start = 1):
        print(f"\nTesting sample {i}")

        
        normalized = normalizePose(test_kp)
        assert normalized.shape == (12, 3), f"Expected (12,3) got {normalized.shape}"
        print("  normalizePose: OK")

        jc = computeJc(normalized)
        assert jc.shape == (36,), f"Expected (36,) got {jc.shape}"
        print("  computeJc: OK")

        jjd = computeJjd(normalized)
        assert jjd.shape == (66,), f"Expected (66,) got {jjd.shape}"
        assert np.all(jjd >= 0), "All distances should be non negative"
        print("  computeJjd: OK")

        jjo, jjoFlat = computeJjo(normalized)
        assert jjo.shape == (66, 3), f"Expected (66,3) got {jjo.shape}"
        assert jjoFlat.shape == (198,), f"Expected (198,) got {jjoFlat.shape}"
        norms = np.linalg.norm(jjo, axis=1)
        assert np.allclose(norms, 1.0), "All orientation vectors should be unit length"
        print("  computeJjo: OK")        

        jld = computeJld(normalized, jjd)
        assert jld.shape == (190,), f"Expected (190,) got {jld.shape}"
        assert np.all(jld >= 0), "All distances should be non negative"
        print("  computeJld: OK")

        lla = computeLla(jjo)
        assert lla.shape == (171,), f"Expected (171,) got {lla.shape}"
        assert np.all(lla >= 0) and np.all(lla <= np.pi), "All angles should be between 0 and pi"
        print("  computeLla: OK")

        jpd = computeJpd(normalized, jjo)
        assert jpd.shape == (28,), f"Expected (28,) got {jpd.shape}"
        print("  computeJpd: OK")

        lpa = computeLpa(normalized, jjo)
        assert lpa.shape == (57,), f"Expected (57,) got {lpa.shape}"
        assert np.all(lpa >= 0) and np.all(lpa <= np.pi), "All angles should be between 0 and pi"
        print("  computeLpa: OK")

        ppa = computePpa(normalized, jjo)
        assert ppa.shape == (3,), f"Expected (3,) got {ppa.shape}"
        assert np.all(ppa >= 0) and np.all(ppa <= np.pi), "All angles should be between 0 and pi"
        print("  computePpa: OK")

        gpd = computeGpd(test_kp)
        assert gpd.shape == (749,), f"Expected (749,) got {gpd.shape}"
        assert not np.any(np.isnan(gpd)), "GPD vector should not contain NaN values"
        assert not np.any(np.isinf(gpd)), "GPD vector should not contain inf values"
        print("  computeGpd: OK")

        print("\nAll tests passed")

    return 


if __name__ == "__main__":
    testGpd() #all tests pass
