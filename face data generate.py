import cv2
import os
import numpy as np
import sys


c= 0
X,y=[],[]
for root ,dirs ,files in os.walk('C:\\Users\\lxd123456\\PycharmProjects\\pythonProject\\my_face'):
    for dir in dirs:


                im =cv2.imread(os.path.join(root,dir),cv2.IMREAD_GRAYSCALE)
                print(os.path.join(root,dir))
                im =cv2.resize(im,(200,200))
                X.append(np.asarray(im,dtype=np.uint8))
                y.append(c)
                c=c+1
    for file in files :
        print(os.path.join(root,file))

