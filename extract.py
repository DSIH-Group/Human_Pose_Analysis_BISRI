##will extract neccessary info from annotation to construct frame by frame pose set

import json
from collections import defaultdict


try:
    with open('camma_mvor_2018.json') as f:
        data = json.load(f)
except FileNotFoundError:
    print("file not found")

except:
    print("other error")

annot_3D = data["annotations3D"] #3D pose annotations

#--------group poses per frame ------------
'''
relevant info from annotation: image_ids, person_id, keypoints3D <-- for each annotation we will extract this info

poses_per_frame = {
    frame_1: [ {pose1}, ... , {poseN}],
    ...
    frame_N: [ {pose1}, ... , {poseN}],
}
'''

 
poses_per_frame = defaultdict(list) 

for annot in annot_3D:
    frame = annot['image_ids']
    person = annot['person_id']
    keypoints = annot['keypoints3D']


    poses_per_frame[frame].append({
        'person': person,
        'keypoints': keypoints
    })

# print(len(poses_per_frame)) 629 frames

#print(list(poses_per_frame.values())[0])
'''
multiPersonFrame = 0
for pose in list(poses_per_frame.values()):
    if len(pose)>1:
        multiPersonFrame +=1

print(multiPersonFrame) #313 frames with more than on person
'''


#------based on poses per frame, group operations--------
'''
operations = {
    day_x: {operation_x: [frames]}

}

'''

opRanges = { #span of operations
    'day1': {
        (0, 56)
    },
    'day2': {
        (0, 117),
        (118, 232),
        (233, 234),
        (235, 245),
        (246, 257),
        (258, 270),
        (271, 278),
        (279, 293),
        (294, 321),
        (322, 324),
        (325, 329)
    },
    'day3': {
        (0, 7),
        (8, 28),
        (29, 58),
        (59, 84),
        (85, 124),
        (125, 138),
        (139, 160),
        (161, 172),
        (173, 181),
        (182, 196),
        (197, 217)
    },
    'day4': {
        (0, 75),
        (76, 116),
        (117, 121)
    }
}

operations = defaultdict(lambda: defaultdict(list)) 
def getIndex(imageId): #extracts image index from id 
    return int(imageId.split('_')[0][-6:]) #ex: 10010000002_10020000002_10030000002 --> 000002 --> 2)

def getDay(imageId): #extracts day from id
    day = int(imageId.split('_')[0][0])

    match day:
        case 1:
            return 'day1'
        
        case 2:
            return 'day2'
        
        case 3:
            return 'day3'
        
        case 4:
            return 'day4'
    

#iterating through frames and mapping them to their respective operations 
for frameId, poses in poses_per_frame.items():
    dayString = getDay(frameId)
    frameIdx = getIndex(frameId)


    for opIdx, (opStart, opEnd) in enumerate(opRanges[dayString], start =1):
        
        if opStart <= frameIdx <= opEnd:
            opString = f"operation_{opIdx}"
            operations[dayString][opString].append({
                'frameIdx': frameIdx,
                'poses': poses 
            })
            break #make sure it only maps to one operation

#sort to put frames in order
for day in operations:
    for op in operations[day]:
        operations[day][op].sort(key=lambda f: f['frameIdx'])
    print(operations[day]['operation_1'])



        

