import cv2
import numpy as np
import math
import argparse
from os import walk
from os import path
import sys
import time
import Histo
import Matches

Images_DataSet = list()
DataSet = "Logo/SurPhotos/"
for (repertoire, sousRepertoires, fichiers) in walk(DataSet):
    for nameFile in fichiers:
        if nameFile.split('.')[1] == "jpg" or nameFile.split('.')[1] == "png":
            pathImg = repertoire + '/' + nameFile
            Images_DataSet.append(pathImg)
a = time.time()
output = list()
for image in Images_DataSet:
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    coca_cascade = cv2.CascadeClassifier('cascade_clasifier/trained_classifier/coca_cascade_15_train.xml')
    adidas_cascade = cv2.CascadeClassifier('cascade_clasifier/trained_classifier/adadas_cascade_15_train.xml')
    nike_cascade = cv2.CascadeClassifier('cascade_clasifier/trained_classifier/nike_cascade_15_train.xml')
    if (img.shape[0] < 1080 and img.shape[1] < 1920) or (
            img.shape[0] > 1500 and img.shape[1] > 2500):
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coca = coca_cascade.detectMultiScale(gray,
                            scaleFactor= 1.1,
                            minNeighbors=5,
                            minSize= (30,30), #(60, 20),
                            flags = cv2.CASCADE_SCALE_IMAGE)
    adidas = adidas_cascade.detectMultiScale(gray,
                            scaleFactor= 1.1,
                            minNeighbors=5,
                            minSize= (30,30), #(60, 20),
                            flags = cv2.CASCADE_SCALE_IMAGE)
    nike = nike_cascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(30, 30),  # (60, 20),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in coca:
        print('ok')
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for (x, y, w, h) in adidas:
        print('ok')
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,0 , 0), 2)
    for (x, y, w, h) in nike:
        print('ok')
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    output.append(img)

execution_time = format(time.time() - a, '.2f')
print("temps d'execution : " + str(execution_time) + "sec")
for img in output:
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
