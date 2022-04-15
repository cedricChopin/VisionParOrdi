import numpy as np
import cv2
img = cv2.imread('Logo/SurPhotos/Coca 2.jpg')

cv2.imshow('img', img)
cv2.waitKey()

coca_cascade = cv2.CascadeClassifier('cascade_clasifier/trained_classifier/coca_cascade_10_train.xml')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

coca = coca_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in coca:
    print('ok')
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey()

# def detect(faceCascade, gray_, scaleFactor_=1.1, minNeighbors=5):
#     faces = faceCascade.detectMultiScale(gray_,
#                                          scaleFactor=scaleFactor_,
#                                          minNeighbors=5,
#                                          minSize=(30, 30),  # (60, 20),
#                                          flags=cv2.CASCADE_SCALE_IMAGE
#                                          )
#     return faces
#
#
# def DetectAndShow(imgfolder='NegFromAds/coca cola advertisements/'):
#     cokelogo_cascade = "./cascade4/cokelogoorigfullds.xml"
#     cokecascade = cv2.CascadeClassifier(cokelogo_cascade)
#     for i in os.listdir(imgfolder):
#         filepath = imgfolder + i
#         img = cv2.imread(filepath)
#
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         cokelogos = detect(cokecascade, gray, 1.25, 6)
#         for (x, y, w, h) in cokelogos:
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.imshow('positive samples', img)
#         k = 0xFF & cv2.waitKey(0)
#         if k == 27:  # q to exit
#             break
