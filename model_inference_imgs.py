import cv2
from ultralytics import YOLO
from Equip_detct import equi_detct
import time
from model_predict import face_rec

#setting a default color which will be changed if condtion is not fulfilled
color=(0,255,0)
#setting default name just like color
name='Properly_Equipped'
#getting time just to calculate hiw much time was taken to calculate a frame(starting time)
starttim=time.time()
#reading img
frame=cv2.imread("F:/Proj_SA/Project/SirAli_Project/DEMO/5.png")
#importing model to get person
model=YOLO("F:\Proj_SA\Project\SirAli_Project\DEMO\person.pt")
#making prediction using the model
person = model.predict(frame)
#if a person is detected
if person:
       #it will iterate through all of them
       for p in person:
             #get its box coordinates
             boxes=p.boxes
             #get shape of results to know total no of person detcted
             shape=boxes.shape[0]
             #iterating through all person
             for k in range(shape):
              #accesing their box attribute
              for box in boxes:
                #creating alist to get missing equipment
                missing_ppe=[]
                #getting coordinates of boxes of each person detcted one at a time
                x1,y1,x2,y2=boxes.xyxy[k]
                #type casting
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                #getting the portion of frame in which the person was detcted 
                img=frame[y1-50:y2+50,x1-50:x2+50]
                #getting the list from equipment_detect to jnow which items where detected
                ppe_List=[]
                #passing only the area in which person was detected to detect equipment so that if the equipemnt is present elsewhere so it wont get detected 
                ppe_list=equi_detct(img)
                print(ppe_list)#just for checking the list can be removed
               #iterating through the list to know which equipment is missing
                for p in ppe_list:
                  # adding equipment to the list and changes its color to red if any equip is missing
                  if (p=="NO-Hardhat")or(p=="NO-Safety Vest")or(p=="no-glasses"): 
                     missing_ppe.append(p)     
                     color=(0,0,255)
                     #passing the image for facial recognition 
                     name=face_rec(img)
                     if name=="Face not Detected":
                       name="unknown"
                
    
              
              print(missing_ppe)     #just for checking the list can be removed
              #creating rectgle to highlight the person
              cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
              #putting  text over rectangle
              y2=y1-50
              cv2.putText(frame,name,(x1,y2),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)
              j=20
   
              for i in missing_ppe:
                cv2.putText(frame,i,(x1,y2+j),cv2.FONT_HERSHEY_COMPLEX,0.55,color,2)
                j=j+20
  #display frame
cv2.imshow("frame",frame)
cv2.waitKey(0)