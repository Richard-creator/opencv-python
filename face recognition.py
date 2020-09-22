import cv2
import os
import numpy as np


def read_images():
 c= 0
 X,y=[],[]
 for root ,dirs ,files in os.walk('C:\\Users\\lxd123456\\PycharmProjects\\pythonProject\\my_face'):

    for file in files:
            try:


                im =cv2.imread(os.path.join(root,file),cv2.IMREAD_GRAYSCALE)
                print(os.path.join(root,file))
                im =cv2.resize(im,(200,200))
                X.append(np.asarray(im,dtype=np.uint8))
                y.append(c)
                c=c+1
            except :
                print('Unexpected  error',sys.exc_info()[0])

 return [X,y]



read_images()
[X,y]=read_images()
y = np.asarray(y,dtype=np.int32)

model = cv2.face.EigenFaceRecognizer_create()
model.train(np.asarray(X),np.asarray(y))
import sys
import video
try:
        fn = sys.argv[1]
except IndexError:
        fn = 0
cap = video.create_capture(fn)

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while (cap.isOpened()):

    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:

            roi =gray[x:x+w,y:y+h]
            try:
                roi =cv2.resize(roi,(200,200),interpolation= cv2.INTER_LINEAR)
                params = model.predict(roi)
                print("label:%s,confidence:%.2f"%(params[0],params[1]))
                print("w:%s h:%s"%(w,h))
                if (params[1]>8000)&(w>100)&(h>100) :
                  cv2.putText(img,'lixinda',(x,y-20),cv2.FONT_HERSHEY_COMPLEX,1,255,2)
                  cv2.rectangle(img, (x, y), (x + w, y + h), (123, 123, 255), 3)
            except:
                continue
    cv2.imshow('detector',img)

    if cv2.waitKey(1)==27:
            break
cap.release()
cv2.destroyAllWindows()


