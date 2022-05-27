#!/usr/bin/env python
# coding: utf-8

# In[8]:


pip install opencv-python


# In[23]:


import cv2


# In[62]:


facebook = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
pathf = "C:\\Users\\CCE\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"
facebook.load(pathf)

img = cv2.imread("ddd.jpg")
q = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


face =  facebook.detectMultiScale(
    q,
    scaleFactor=1.12,
    minNeighbors=3,
    minSize=(10,10)
)
for (x,y,w,h) in face:
    roi_gray = q[y:y+h, x:x+w]
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# cv2.imwrite("ddd4.jpg", img)
print(len(face))


# In[41]:


print(q)


# In[14]:


cv2.imshow("v",q)
cv2.waitKey()


# In[4]:


import cv2
import numpy as np

lst = []

lst.append(cv2.imread("c1.jpg",cv2.IMREAD_GRAYSCALE))
lst.append(cv2.imread("c2.jpg",cv2.IMREAD_GRAYSCALE))
lst.append(cv2.imread("c3.jpg",cv2.IMREAD_GRAYSCALE))
lst.append(cv2.imread("d1.jpg",cv2.IMREAD_GRAYSCALE))
lst.append(cv2.imread("d2.jpg",cv2.IMREAD_GRAYSCALE))
lst.append(cv2.imread("d3.jpg",cv2.IMREAD_GRAYSCALE))

labels = [0,0,0,1,1,1]

recongnizer =  cv2.face.LBPHFaceRecognizer_create()
recongnizer.train(lst,np.array(labels))

p_img = cv2.imread("c3.jpg",cv2.IMREAD_GRAYSCALE)

label,confidence = recongnizer.predict(p_img)

print("label=",label)
print("confidence=",confidence)


# In[10]:





# In[ ]:




