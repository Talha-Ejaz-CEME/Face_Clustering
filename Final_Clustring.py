import glob
from sklearn.cluster import DBSCAN
import hdbscan
import numpy as np
import pickle
import cv2
import shutil
import os
import face_recognition


# In[3]:

## data path
root_path=r"dataset"
Paths=[]
# path for all files
for name in glob.glob(root_path+'/*'):
        #print(name.split("Files\\")[-1])        
        Paths.append(name)


# In[4]:


len(Paths)


# In[5]:


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
  


# In[6]:


data = np.array(data)

encodings = [d["encoding"] for d in data]
print(len(encodings))

print(type(encodings))


# In[7]:


print("DBSCAN")


## adjust min_samples value
clt = DBSCAN(eps=0.5,metric="euclidean", n_jobs=2,min_samples=3)
clt.fit(encodings)
#clt.labels_
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
#numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

ALL_Clusters_DB=[]

for labelID in labelIDs:

    # find all the indexes into the 'data' array that belong to the
    # current label ID, then randomly sample a maximum of 25 index from the set
    print("[INFO] faces for face ID: {}".format(labelID))
    idxs = np.where(clt.labels_ == labelID)[0]
    #idxs = np.random.choice(idxs, size=min(25, len(idxs)),replace=False)
    # initialize the list of faces to include in the montage
    faces = []
    # loop over the sampled indexes
    
    Cluster=[]
    
    for i in idxs:
        Cluster.append(data[i]["imagePath"])

    ALL_Clusters_DB.append(Cluster)
    title = "Face ID #{}".format(labelID)
    title = "Unknown Faces" if labelID == -1 else title
    print(Cluster)
    

    
#for list in ALL_Clusters_DB:
#   print(list)


print("HDBSCAN")

# In[21]:

## adjust min_cluster_size
clusterer = hdbscan.HDBSCAN(min_cluster_size=3,gen_min_span_tree=True)
clusterer.fit(encodings)
labelIDs = np.unique(clusterer.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
#numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

ALL_Clusters_HDB=[]

for labelID in labelIDs:
    # find all the indexes into the 'data' array that belong to the
    # current label ID, then randomly sample a maximum of 25 index from the set
    print("[INFO] faces for face ID: {}".format(labelID))
    idxs = np.where(clusterer.labels_ == labelID)[0]
    #idxs = np.random.choice(idxs, size=min(25, len(idxs)),replace=False)
    # initialize the list of faces to include in the montage
    Cluster=[]
    
    for i in idxs:
        Cluster.append(data[i]["imagePath"])

    ALL_Clusters_HDB.append(Cluster)
    title = "Face ID #{}".format(labelID)
    title = "Unknown Faces" if labelID == -1 else title
    print(Cluster)
    
#for list in ALL_Clusters_HDB:
#   print(list)




