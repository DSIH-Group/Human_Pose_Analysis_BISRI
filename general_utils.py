
import numpy as np
#-------function useful across whole project--------
#converting flat array into 12x3
def convertKeypoints(keypoints):
    coords = []

    #removing mystery coordinate
    for i, coord in enumerate(keypoints):
        if i % 4 != 3:
            coords.append(coord)

    # reshape to 12x3
    joints = []
    for i in range(0, len(coords), 3):
        joints.append(coords[i:i+3])

    return joints

def printKp(keypoints): #helper to print keypoints joint by joint
    for i in range(len(keypoints)):
        print(f"{i+1}: {keypoints[i]}")

def printFrame(frame): # helpers to print operations in a cleaner manner (for visuals)
    print(f"Frame: {frame['frameIdx']}")
    print("---------Poses----------")
    for pose in frame["poses"]:
        print(f"Person {pose["person"]}")
        print("Keypoints:")
        printKp(pose["keypoints"])
        print("\n")


def printOp(operation): #cleanely print operation, frame by frame 
    for frame in operation:
        printFrame(frame)

def validateShape(cleanedAnnots):
    all_valid = True
    for day in cleanedAnnots:
        for operation in cleanedAnnots[day]:
            for frame in cleanedAnnots[day][operation]:
                for pose in frame['poses']:
                    kp = pose['keypoints']
                    if not isinstance(kp, np.ndarray):
                        print(f"Not numpy array: day={day}, op={operation}, frame={frame['frameIdx']}, person={pose['person']}")
                        all_valid = False
                    elif kp.shape != (12, 3):
                        print(f"Wrong shape {kp.shape}: day={day}, op={operation}, frame={frame['frameIdx']}, person={pose['person']}")
                        all_valid = False

    if all_valid:
        print("All keypoints valid: numpy arrays of shape (12, 3)")
