from ultralytics import YOLO
import cv2
import math


# loading models for detection of ppe 
equi=YOLO('F:\\Proj_SA\\Project\\SirAli_Project\\Model\\ultralytics\\runs\\detect\\custom2\\weights\\best.pt') 
glass=YOLO('F:\Proj_SA\Project\SirAli_Project\DEMO\glasses.pt')

#def color as global var
def equi_detct(frame):
   color=(0,0,0)
   #img=cv2.resize(img,(800,1000))
   
# list of detected equipments to be returned at the end
   equi_detct=[]  
   try:
    #detecting ppe in the frame
    res=equi.predict(frame,conf=0.45)
   except:
     #incase frame wasnt passed. to handle the error and return a blank list it occurs after the last frame of the video has been processed but cap is still opened
     equi_detct.append('no__prdict__due__to__zero__error')
     return equi_detct
    #for detection of glasses
   glassres=glass.predict(frame,conf=0.45)
    # defining classes in order to access label using index returned by the model  
   classes=['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']
   #iterating through array returned by the model and accessing the bounding boxes to make predictions   
   for r in res:
     #accessing boxes from each prediction made
     boxes=r.boxes
     #iterating through boxes 
     for box in boxes:
      #getting the coordinates of boxes
      x1,y1,x2,y2=box.xyxy[0]
      #explicitly parsing their datatype
      x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
      #calculating their widthand height
      w,h=x2-x1,y2-y1
      #accessing index of label returned
      cls =int(box.cls[0])
      #getting the conf of the object  
      conf = math.ceil((box.conf[0] * 100)) / 100
      cn=classes[cls] #getting current class using index 
      print(cls)
      #seeting condiiton for conf
      if conf>0.5:
       #appending the detected equipment in equidetct
       if cn=='Hardhat':
         equi_detct.append(cn)
         color=(0,255,0)
      if cn=='Safety Vest':
         equi_detct.append(cn)
         color=(0,255,0) 
      if cn=='NO-Hardhat' :
         equi_detct.append(cn)
         color=(0,0,255)
      if cn=='NO-Safety Vest':
         equi_detct.append(cn)
         color=(0,0,255)   
      else:
         continue
      #image = cv2.putText(frame, f'{classes[cls]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255 ,255), 2, cv2.LINE_AA)
     # cv2.rectangle(frame, (x1, y1), (x2, y2),color, 3)
   #acceing the bounding box of glasses and performing same operations as above
   for g in glassres:
        if g:
         boxes=g.boxes
         for box in boxes:
           clas=(box.cls[0])
           equi_detct.append("glasses")
        else:
            equi_detct.append("no-glasses")
   return equi_detct