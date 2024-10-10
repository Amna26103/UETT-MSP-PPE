###This file is to get face embedding for facial area  extracted in face1

#pip install keras-facenet
import cv2
from keras_facenet import FaceNet
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'# just to turn off warnings other than those which are more likely to cause errors
##getting facial embedding and saving them to a file
#loading the model to get facial embeddings 
embedder=FaceNet()
def get_embeddings(face_img):
    #defining img's datatype 
    face_img=face_img.astype('float32')
    #expandiing dimension of img
    face_img=np.expand_dims(face_img,axis=0)
    #getting  embeeding from the frame passed to the function
    yhat=embedder.embeddings(face_img)
    #returning embbedings  of the frame
    return yhat[0]

