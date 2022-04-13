import cv2
import numpy as np
import math
import argparse
from os import walk
import sys
import time

a = time.time()
#Prend une image en entrée, ainsi que des bords si rectangle
#retourne les histogrammes des channels B, G et R respectivements
def histo(im, bounds = None):
    mask = None
    if bounds != None:
        mask = np.zeros(im.shape[:2])
        cv2.rectangle(mask, bounds[0], bounds[2], (255,255,255), -1)

    histB = cv2.calcHist([im],[0], mask,[8],[0,256])
    histG = cv2.calcHist([im],[1], mask,[8],[0,256])
    histR = cv2.calcHist([im],[2], mask,[8],[0,256])

    return histB, histG, histR
    #return histB

#Prend les match et les histogrammes associés
#Retourne le pourcentage de correspondance selon la distance et les couleurs
'''def matchRate(matches, histosIm, histosLogo):
    minDist = 99

    for m in matches:
        if m.distance < minDist : minDist = m.distance

    correspondance = 1-minDist/100

    histoBDiff = 0;
    histoGDiff = 0;
    histoRDiff = 0;

    (histoImB,histoImG, histoImR) = histosIm
    (histoLogoB,histoLogoG, histoLogoR) = histosIm

    for i in len(histoImB):
        histoBDiff += abs(histoImB[i] - histoLogoB[i])
        histoGDiff += abs(histoImG[i] - histoLogoG[i])
        histoRDiff += abs(histoImR[i] - histoLogoR[i])

    histoDiff = (histoBDiff/len(histoImB) + histoGDiff/len(histoImG) + histoRDiff/len(histoImR))/3

    return (correspondance, histoDiff)
'''
def matchRate(histosIm, histosLogo):
    histoBDiff = 0;
    histoGDiff = 0;
    histoRDiff = 0;

    histoImB, histoImG, histoImR = histosIm
    histoLogoB,histoLogoG, histoLogoR = histosLogo

    size = len(histoImB)

    for i in range(size):
        histoBDiff += abs(histoImB[i] - histoLogoB[i])
        histoGDiff += abs(histoImG[i] - histoLogoG[i])
        histoRDiff += abs(histoImR[i] - histoLogoR[i])

    print(histoBDiff/size)
    print(histoGDiff/size)
    print(histoRDiff/size)

    histoDiff = ((histoBDiff/size) + (histoGDiff/size) + (histoRDiff/size))/3
    return histoDiff


def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


for i in progressbar(range(100), "Computing : ", 40):
    time.sleep(0.01)
    execution_time = format(time.time() - a,'.2f')
    #print(execution_time)

test = cv2.imread("./Logo/test.jpg")
test = cv2.resize(test, (int(test.shape[1]/2),int(test.shape[0]/2)))

test2 = cv2.imread("./Logo/test2.jpg")
test2 = cv2.resize(test2, (int(test2.shape[1]/2),int(test2.shape[0]/2)))

h = histo(test)
h2 = histo(test2)

r = matchRate(h, h2)
print(r)