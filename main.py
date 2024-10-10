import cv2
from ultralytics import YOLO
from Equip_detct import equi_detct
import time
from face_recognition import face_rec


color=(0,255,0)
name='Properly_Equipped'
starttim=time.time()
cap = cv2.VideoCapture("F:\Proj_SA\Project\SirAli_Project\DEMO\Videos_demo/3.mp4")
model=YOLO("F:\Proj_SA\Project\SirAli_Project\DEMO\person.pt")

frameno=0
while True:
  ret,frame=cap.read()
  frame=cv2.resize(frame,(640,700))
  if not ret:
             print("Check path of video in VideoCapture!!!")
  frameno=frameno+1
  person = model.predict(frame)
  if person:
       for p in person:
             boxes=p.boxes
             shape=boxes.shape[0]
             for k in range(shape):
              for box in boxes:
                missing_ppe=[]
                x1,y1,x2,y2=boxes.xyxy[k]
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                img=frame[y1-50:y2+50,x1-50:x2+50]
                ppe_List=[]
                ppe_list=equi_detct(img)
                print(ppe_list)#just for checking the list can be removed
                for p in ppe_list:
                  if (p=="NO-Hardhat")or(p=="NO-Safety Vest")or(p=="no-glasses"): 
                     missing_ppe.append(p)     
                     color=(0,0,255)
                     name=face_rec(img)
                  
              print("-----------------------------mossing eqyi\n")
              print(missing_ppe)     #just for checking the list can be removed
              cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
              y2=y1-100
              cv2.putText(frame,name,(x1,y2),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,2)
              j=20
              for i in missing_ppe:
                cv2.putText(frame,i,(x1,y2+j),cv2.FONT_HERSHEY_COMPLEX,0.55,color,2)
                j=j+20
  if cv2.waitKey(1)&0xFF==ord('q'):
           break
       
  cv2.imshow('frame',frame)
times=time.time()-starttim
print(frameno)
print(times)
cap.release()
cv2.destroyAllWindows()