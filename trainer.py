import os
import cv2
import numpy as np
from PIL import Image
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
path='C:\\Users\\HP\\Desktop\\data\\dataSet'
def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        print(ID)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return IDs,faces 
IDs,faces=getImagesWithID(path)
face_recognizer.train(faces,np.array(IDs))
face_recognizer.save("C:\\Users\\HP\\Desktop\\trainingdata\\trainingData.yml")
cv2.destroyAllWindows()
