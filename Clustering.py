from sklearn.cluster import DBSCAN
import hdbscan
from imutils import build_montages
import numpy as np
import pickle
import cv2
import shutil
import os
import face_recognition

root_path=r"dataset"
Paths=[]
# path for all files
for name in glob.glob(root_path+'/*'):
        #print(name.split("Files\\")[-1])        
        Paths.append(name)



data = []
for path in Paths:

    image = cv2.imread(path)

    # ocnverting image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    size=np.shape(image)
    
    if (size[0]>1500 or size[1]>1500):
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    elif (size[0]>750 or size[1]>750):
        image = cv2.resize(image, (0, 0), fx=0.75, fy=0.75)


    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(image,model='cnn')

    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(image, boxes)

    # build a dictionary of the image path, bounding box location,
    # and facial encodings for the current image
    d= [{"imagePath":path, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
    data.extend(d)
    


encodings = [d["encoding"] for d in data]
print(len(encodings))


print("DBSCAN")


clt = DBSCAN(eps=0.5,metric="euclidean", n_jobs=2,min_samples=3)
clt.fit(encodings)
#clt.labels_
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
#numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

for labelID in labelIDs:
    
    path = os.path.join('DataSet/Output_DBSCAN', str(labelID))
    os.mkdir(path)
    
    # find all the indexes into the 'data' array that belong to the
    # current label ID, then randomly sample a maximum of 25 index from the set
    print("[INFO] faces for face ID: {}".format(labelID))
    idxs = np.where(clt.labels_ == labelID)[0]
    #idxs = np.random.choice(idxs, size=min(25, len(idxs)),replace=False)
    # initialize the list of faces to include in the montage
    faces = []
    # loop over the sampled indexes
    for i in idxs:
        # load the input image and extract the face ROI
        image = cv2.imread(data[i]["imagePath"])
        (top, right, bottom, left) = data[i]["loc"]
        
        size=np.shape(image)
    
        if (size[0]>1500 or size[1]>1500):
            (top, right, bottom, left) = (top*2, right*2, bottom*2, left*2)
        elif (size[0]>750 or size[1]>750):
            (top, right, bottom, left) = (int(top*1.34), int(right*1.34), int(bottom*1.34), int(left*1.34))
            
        face = image[top:bottom, left:right]

        # force resize the face ROI to 96x96 and then add it to the
        # faces montage list
        face = cv2.resize(face, (96,96))
        faces.append(face)
        
        # Source path
        source = data[i]["imagePath"]
        
        # Destination path
        destination = os.path.join(path,data[i]["imagePath"].split("dataset\\")[-1])
        
        shutil.copy(source, destination)

    # create a montage using 96x96 "tiles" with 5 rows and 5 columns
    montage = build_montages(faces, (96,96), (5,5))[0]

    # show the output montage
    title = "Face ID #{}".format(labelID)
    title = "Unknown Faces" if labelID == -1 else title
    #cv2.imshow(title,montage)
    #cv2.waitKey(0)
    cv2.imwrite(os.path.join('DataSet/Output_DBSCAN',title + '.jpg'),montage)
    


print("HDBSCAN")

    
clusterer = hdbscan.HDBSCAN(min_cluster_size=3,gen_min_span_tree=True)
clusterer.fit(encodings)
labelIDs = np.unique(clusterer.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
#numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

for labelID in labelIDs:
    path = os.path.join('DataSet/Output_HDBSCAN', str(labelID))
    os.mkdir(path)
    # find all the indexes into the 'data' array that belong to the
    # current label ID, then randomly sample a maximum of 25 index from the set
    print("[INFO] faces for face ID: {}".format(labelID))
    idxs = np.where(clusterer.labels_ == labelID)[0]
    #idxs = np.random.choice(idxs, size=min(25, len(idxs)),replace=False)

    # initialize the list of faces to include in the montage
    faces = []
    # loop over the sampled indexes
    for i in idxs:
        # load the input image and extract the face ROI
        image = cv2.imread(data[i]["imagePath"])
        (top, right, bottom, left) = data[i]["loc"]
        
        size=np.shape(image)
    
        if (size[0]>1500 or size[1]>1500):
            (top, right, bottom, left) = (top*2, right*2, bottom*2, left*2)
        elif (size[0]>750 or size[1]>750):
            (top, right, bottom, left) = (int(top*1.34), int(right*1.34), int(bottom*1.34), int(left*1.34))
            
                                          
        face = image[top:bottom, left:right]
        
        # Source path
        source = data[i]["imagePath"]
        
        # Destination path
        destination = os.path.join(path,data[i]["imagePath"].split("dataset\\")[-1])
        
        shutil.copy(source, destination)

        # force resize the face ROI to 96x96 and then add it to the
        # faces montage list
        face = cv2.resize(face, (96,96))
        faces.append(face)

    # create a montage using 96x96 "tiles" with 5 rows and 5 columns
    montage = build_montages(faces, (96,96), (5,5))[0]

    # show the output montage
    title = "Face ID #{}".format(labelID)
    title = "Unknown Faces" if labelID == -1 else title
    #cv2.imshow(title,montage)
    #cv2.waitKey(0)
    cv2.imwrite(os.path.join('DataSet/Output_HDBSCAN',title + '.jpg'),montage)


import sys
sys.exit()
