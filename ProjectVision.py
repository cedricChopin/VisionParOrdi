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


# Fonction permettant de trouver les logos dans une image
def Draw(Images, name, dataset):
    img_lst = list()  # Liste des images ouvertes
    img_color = list()
    orb_img = cv2.ORB_create(nfeatures=10000, edgeThreshold=1)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    outputImg = []

    if Images is not None:
        for image in Images:
            img = cv2.imread(image, cv2.IMREAD_COLOR)
            img_lst.append(img)
            img_color.append(list(np.random.choice(range(256), size=3)))

    des_list = list()
    kp_lst = list()
    for image in img_lst:
        kp, des = orb_img.detectAndCompute(image, None)
        des_list.append(des)
        kp_lst.append(kp)
    for img in progressbar(dataset, "Analysing images ... : ", 60):
        img_dataset = cv2.imread(img, cv2.IMREAD_COLOR)
        if (img_dataset.shape[0] < 1080 and img_dataset.shape[1] < 1920) or (
                img_dataset.shape[0] > 1500 and img_dataset.shape[1] > 2500):
            img_dataset = cv2.resize(img_dataset, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        cleanCopy = img_dataset
        kp2, des2 = orb_img.detectAndCompute(img_dataset, None)
        # Partie Homography
        # Itération sur chaque image
        bestMatche = list()
        bestCor = 0
        bestLogo = ''
        #print("______________________________________")
        for index_kp in range(len(kp_lst)):
            goodpoints = Matches.GoodPoints(des_list[index_kp], des2, bf)
            cor = Histo.matchRateMatches(goodpoints)
            if cor > bestCor:
                bestCor = cor
                bestLogo = name[index_kp]
            #print("cor: " + str(cor) + "; name : " + name[index_kp])
            # print("Nb good : " + str(len(goodpoints)) + "; name : " + name[index_kp])
            if len(goodpoints) > 25:
                sch_pts = np.float32([kp_lst[index_kp][m.queryIdx].pt for m in goodpoints]).reshape(-1, 1, 2)
                img_pts = np.float32([kp2[m.trainIdx].pt for m in goodpoints]).reshape(-1, 1, 2)
                matrix, mask = cv2.findHomography(sch_pts, img_pts, cv2.RANSAC, 5.0)

                if matrix is not None:
                    pts = np.float32([[0, 0],
                                      [0, img_lst[index_kp].shape[0] - 1],
                                      [img_lst[index_kp].shape[1] - 1, img_lst[index_kp].shape[0] - 1],
                                      [img_lst[index_kp].shape[1] - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, matrix)
                    rect = cv2.minAreaRect(dst)

                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    col = img_color[index_kp]
                    # Filtre par angle pour prendre des régions cohérentes
                    angle_bas = Matches.getAngle(dst[0][0], dst[1][0], dst[2][0])
                    angle_haut = Matches.getAngle(pts[2][0], pts[3][0], pts[0][0])
                    if (70 < angle_bas < 110) and (70 < angle_haut < 110):
                        xmin = min([pt[0] for pt in box])
                        xmax = max([pt[0] for pt in box])
                        ymin = min([pt[1] for pt in box])
                        ymax = max([pt[1] for pt in box])
                        boundingBox = (ymin, ymax, xmin, xmax)
                        # Calcul de l'histogramme de l'image trouvée et du logo à comparer
                        himg = Histo.histo(cleanCopy, boundingBox)
                        imgResized = cv2.resize(img_lst[index_kp], (int(ymax - ymin), int(xmax - xmin)))
                        hLogo = Histo.histo(imgResized)

                        # Application de flou gaussien sur l'histogramme
                        himg = Histo.gaussianblur(himg)
                        hLogo = Histo.gaussianblur(hLogo)
                        # Passage de l'histogramme en 8x256
                        himg = Histo.cutHisto(himg)
                        hLogo = Histo.cutHisto(hLogo)

                        # Récupération du pourcentage de différence entre deux histogrammes
                        r = Histo.matchRateHisto(himg, hLogo, imgResized.shape[0] * imgResized.shape[1])
                        bestMatche.append((r, index_kp, col, box))
                        #print(str((1 - r) * 100) + " de difference")
        bottomLeftCornerOfTextCompa = (int(50), int(50))
        bottomLeftCornerOfTextColor = (int(50), int(100))
        bottomLeftCornerOfTextLogo = (int(50), int(150))

        fontScale = 2
        fontColor = (0, 255, 0)
        thickness = 3
        lineType = 2
        if len(bestMatche) > 0:
            best = min([tuple[0] for tuple in bestMatche])
            bestTuple = [tuple for tuple in bestMatche if tuple[0] == best]

            x_offset  = 50
            y_offset = 200
            logoResized = cv2.resize(img_lst[bestTuple[0][1]],
                                     (int(img_dataset.shape[1] / 10), int(img_dataset.shape[0] / 10)))
            img_dataset[y_offset:y_offset + logoResized.shape[0],
            x_offset:x_offset + logoResized.shape[1]] = logoResized
            cv2.drawContours(img_dataset, [bestTuple[0][3]], 0,
                             [int(bestTuple[0][2][0]), int(bestTuple[0][2][1]), int(bestTuple[0][2][2])], 2)
            img_dataset = cv2.putText(img_dataset, "Pourcentage compatibilite Color : " + str(bestTuple[0][0][0] * 100) + " %",
                                      bottomLeftCornerOfTextColor,
                                      1,
                                      fontScale,
                                      fontColor,
                                      thickness,
                                      lineType)


        img_dataset = cv2.putText(img_dataset, "Pourcentage compatibilite Matches : " + str(bestCor * 100) + " %",
                                  bottomLeftCornerOfTextCompa,
                                  1,
                                  fontScale,
                                  fontColor,
                                  thickness,
                                  lineType)

        img_dataset = cv2.putText(img_dataset, "Logo trouve : " + bestLogo,
                                  bottomLeftCornerOfTextLogo,
                                  1,
                                  fontScale,
                                  fontColor,
                                  thickness,
                                  lineType)
        outputImg.append(img_dataset)
    return outputImg


# Fonction permettant de récuperer les logos ainsi que le dataset
def getImages(Logos, DataSet):
    Images = list()
    Images_DataSet = list()
    name = list()
    # DataSet = "Logo/Dessins/"
    for (root, sousRep, fic) in walk(Logos):
        for dirname in sousRep:
            for (repertoire, sousRepertoires, fichiers) in walk(path.join(root, dirname)):
                for nameFile in fichiers:
                    if nameFile.split('.')[1] == "jpg" or nameFile.split('.')[1] == "png":
                        pathImg = repertoire + '/' + nameFile
                        name.append(dirname)
                        Images.append(pathImg)

    for (repertoire, sousRepertoires, fichiers) in walk(DataSet):
        for nameFile in fichiers:
            if nameFile.split('.')[1] == "jpg" or nameFile.split('.')[1] == "png":
                pathImg = repertoire + '/' + nameFile
                name.append(nameFile.split('.')[0])
                Images_DataSet.append(pathImg)
    return Images, name, Images_DataSet


def main():
    a = time.time()
    Logos = "Logo/Clair/"
    DataSet = "Logo/SurPhotos/"
    Images, name, Images_DataSet = getImages(Logos, DataSet)
    output = Draw(Images, name, Images_DataSet)

    execution_time = format(time.time() - a, '.2f')
    print("temps d'execution : " + str(execution_time) + "sec")

    time.sleep(3)

    for img in output:
        cv2.imshow("results :", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
