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

    histB = cv2.calcHist([im],[0], mask,[256],[0,256])
    histG = cv2.calcHist([im],[1], mask,[256],[0,256])
    histR = cv2.calcHist([im],[2], mask,[256],[0,256])

    return histB, histG, histR

#Prend l'histogramme(ouai bon c'est écrit im mais voilà merde hein)
#Retourne l'histogramme avec le flou gaussien appliqué
def gaussianblur(im):
    convo = {
        (-2, -2): 1,
        (-2, -1): 4,
        (-2, 0): 7,
        (-2, 1): 4,
        (-2, 2): 1,
        (-1, -2): 4,
        (-1, -1): 16,
        (-1, 0): 26,
        (-1, 1): 16,
        (-1, 2): 4,
        (0, -2): 7,
        (0, -1): 26,
        (0, 0): 41,
        (0, 1): 26,
        (0, 2): 7,
        (1, -2): 4,
        (1, -1): 16,
        (1, 0): 26,
        (1, 1): 16,
        (1, 2): 4,
        (2, -2): 1,
        (2, -1): 4,
        (2, 0): 7,
        (2, 1): 4,
        (2, 2): 1
    }

    cop = im

    for x in range(len(im[0])):
        for y in range(len(im)):
            moy = 0

            for i in range(-2, 3):
                for j in range(-2, 3):
                    if (x + i >= 0 and y + j >= 0 and x + i < len(im[0]) and y + j < len(im)):
                        moy += cop[y + j][x + i] * convo[(i, j)] / 273

            cop[y][x] = int(moy)

    return cop

#Prend un histogramme 256x256 et le redécoupe pour avoir qu'un tableau 8x256
#Retourne l'histogramme 8x256
def cutHisto(histo):
    hB = []
    hG = []
    hR = []

    offset = int(len(histo[0]) / 8)
    for i in range(8):
        sumB = 0
        sumG = 0
        sumR = 0
        for j in range(offset):
            sumB += histo[0][int(i*offset + j)]
            sumG += histo[1][int(i*offset + j)]
            sumR += histo[1][int(i*offset + j)]

        hB.append(sumB)
        hG.append(sumG)
        hR.append(sumR)

    return (hB, hG, hR)

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
    histoBDiff = 0
    histoGDiff = 0
    histoRDiff = 0

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


def main():
    test = cv2.imread("./Logo/test.jpg")
    test = cv2.resize(test, (1200, 800))

    test2 = cv2.imread("./Logo/test2.jpg")
    test2 = cv2.resize(test2, (1200, 800))

    test3 = cv2.imread("./Logo/test3.jpg")
    test3 = cv2.resize(test3, (1200, 800))

    test4 = cv2.imread("./Logo/black.png")
    test4 = cv2.resize(test4, (1200, 800))

    test5 = cv2.imread("./Logo/space.jpg")
    test5 = cv2.resize(test5, (1200, 800))

    h = histo(test4)
    h5 = histo(test5)

    h = gaussianblur(h)
    h5 = gaussianblur(h5)

    h = cutHisto(h)
    h5 = cutHisto(h5)

    cv2.imshow("test", test4)
    cv2.waitKey(0)
    cv2.imshow("test2", test5)
    cv2.waitKey(0)

    r = matchRateHisto(h, h5, 1200 * 800)

    print(str((1 - r) * 100) + " de difference")
    execution_time = format(time.time() - a, '.2f')
    print("temps d'execution : " +  str(execution_time))

if __name__ == "__main__":
    main()
