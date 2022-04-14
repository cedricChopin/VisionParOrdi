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
def matchRateMatches(matches):
    minDist = 99

    for m in matches:
        if m.distance < minDist : minDist = m.distance

    correspondance = 1-minDist/100

    return correspondance


#Prend les histogrammes des deux images et les compare
#Retourne la correspondance
def matchRateHisto(histosIm, histosLogo, nbPix):
    histoBDiff = 0;
    histoGDiff = 0;
    histoRDiff = 0;

    histoImB, histoImG, histoImR = histosIm
    histoLogoB,histoLogoG, histoLogoR = histosLogo

    size = len(histoImB)

    for i in range(size):
        histoBDiff += min(histoImB[i], histoLogoB[i])
        histoGDiff += min(histoImG[i], histoLogoG[i])
        histoRDiff += min(histoImR[i], histoLogoR[i])

    return ((histoBDiff/nbPix) + (histoGDiff/nbPix) +(histoRDiff/nbPix))/3

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

test = cv2.imread("./Logo/WHIIITE.png")
test = cv2.resize(test, (1200,800))

test2 = cv2.imread("./Logo/test2.jpg")
test2 = cv2.resize(test2, (1200,800))

test3 = cv2.imread("./Logo/test3.jpg")
test3 = cv2.resize(test3, (1200,800))

test4 = cv2.imread("./Logo/black.png")
test4 = cv2.resize(test4, (1200,800))

test5 = cv2.imread("./Logo/YEEEEEYE.jpg")
test5 = cv2.resize(test5, (1200,800))

h = histo(test)
h2 = histo(test2)
h3 = histo(test3)
h4 = histo(test4)
h5 = histo(test5)


r = matchRateHisto(h, h4, 1200*800)

print(str((1-(r/3*100))*100) + " de difference")