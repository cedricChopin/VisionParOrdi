import numpy as np
import cv2
img = cv2.imread('Logo/SurPhotos/Coca 2.jpg')

cv2.imshow('img', img)
cv2.waitKey()

coca_cascade = cv2.CascadeClassifier('C:/Users/pouyoupy/Documents/GitHub/VisionParOrdi/cascade_clasifier/classifier/cascade.xml')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

coca = coca_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in coca:
    print('ok')
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey()
