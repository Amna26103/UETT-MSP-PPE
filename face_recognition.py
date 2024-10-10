import cv2
import pickle
from embedding_np import get_embeddings
from sklearn.preprocessing import LabelEncoder
import numpy as np
from mtcnn.mtcnn import MTCNN
#the file is to detcet face and recognize them
def face_rec(img):
  #loading the embedding saved during model training
  embeddings=np.load("F:\Face_recog2\Facialembedding.npz")
  #intializing detctor
  detector=MTCNN()
  #getting embedding from file
  Y=embeddings['arr_1']
  #loading encoder
  encoder=LabelEncoder()
  #pasisng the label to encoder
  encoder.fit(Y)
  #intializing labels after transforming into Y
  Y=encoder.transform(Y)
 # opening file to load model 
  with open('F:\Proj_SA\Project\SirAli_Project\DEMO\model2_U.pkl', 'rb') as f:
     model = pickle.load(f)
  #convering the image to rgb for processing
  rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  try:
   #getting coordinates of face detcted
   x,y,w,h=detector.detect_faces(rgb)[0]['box']
   #cropping face from whole frame
   face=img[y:y+h,x:x+w] 
   #resizing t to perform better calculations
   face=cv2.resize(face,(160,160))
   #get embedding of face detcted
   test=get_embeddings(face)
   #reshaping array
   test=[test]
   #predicting face using model
   pred=model.predict(test)
   #performing inverse transformation to get the label
   name=encoder.inverse_transform(pred)
  # printing label
   print(name)
   #returning label
   return name[0]
  except :
     return "Face not Detected"
 