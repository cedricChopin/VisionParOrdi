import cv2
import numpy as np
import math
import argparse
from os import walk
from os import path
import sys
import time
import Histo



def GoodPointsTrain(img, bf):
    matches = bf.knnMatch(img, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def GoodPoints(img, img2, bf):
    matches = bf.knnMatch(img, img2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def Draw(Images, name, dataset):
    img_lst = list()
    img_color = list()
    orb_img = cv2.ORB_create(nfeatures=10000, edgeThreshold=1)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
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

    # bf.add(des_list)
    # bf.train()
    for img in dataset:
        img_dataset = cv2.imread(img, cv2.IMREAD_COLOR)
        if (img_dataset.shape[0] < 1080 and img_dataset.shape[1] < 1920) or (img_dataset.shape[0] > 1500 and img_dataset.shape[1] > 2500):
            img_dataset = cv2.resize(img_dataset, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        cleanCopy = img_dataset
        kp2, des2 = orb_img.detectAndCompute(img_dataset, None)
        # Partie Homography

        logoAlreadyCaptured = list()
        for index_kp in range(len(kp_lst)):
            goodpoints = GoodPoints(des_list[index_kp], des2, bf)
            print("Nb good : " + str(len(goodpoints)) + "; name : " + name[index_kp])
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
                    angle_bas = getAngle(dst[0][0], dst[1][0], dst[2][0])
                    angle_haut = getAngle(pts[2][0], pts[3][0], pts[0][0])
                    if (70 < angle_bas < 110) and (70 < angle_haut < 110):
                        xmin = min([pt[0] for pt in box])
                        xmax = max([pt[0] for pt in box])
                        ymin = min([pt[1] for pt in box])
                        ymax = max([pt[1] for pt in box])
                        boundingBox = (ymin, ymax, xmin, xmax)
                        logoAlreadyCaptured.append(boundingBox)
                        imageCut = cleanCopy[int(ymin):int(ymax), int(xmin):int(xmax)]
                        cv2.imshow("ImageCut", imageCut)
                        x_offset = y_offset = 50
                        logoResized = cv2.resize(img_lst[index_kp],
                                                 (int(img_dataset.shape[1] / 10), int(img_dataset.shape[0] / 10)))
                        img_dataset[y_offset:y_offset + logoResized.shape[0],
                        x_offset:x_offset + logoResized.shape[1]] = logoResized
                        cv2.drawContours(img_dataset, [box], 0, [int(col[0]), int(col[1]), int(col[2])], 2)
                        bottomLeftCornerOfText = (int(dst[0][0][0]), int(dst[0][0][1]))
                        fontScale = 3
                        fontColor = (150, 150, 150)
                        thickness = 3
                        lineType = 2
                        img_dataset = cv2.putText(img_dataset, name[index_kp],
                                                  bottomLeftCornerOfText,
                                                  1,
                                                  fontScale,
                                                  fontColor,
                                                  thickness,
                                                  lineType)


        cv2.imshow("Scene", img_dataset)
        cv2.waitKey(0)

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
    DataSet = "Logo/Sur photos/"
    Images, name, Images_DataSet = getImages(Logos, DataSet)
    Draw(Images, name, Images_DataSet)
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

    h = Histo.histo(test4)
    h5 = Histo.histo(test5)

    h = Histo.gaussianblur(h)
    h5 = Histo.gaussianblur(h5)

    h = Histo.cutHisto(h)
    h5 = Histo.cutHisto(h5)

    cv2.imshow("test", test4)
    cv2.waitKey(0)
    cv2.imshow("test2", test5)
    cv2.waitKey(0)

    r = Histo.matchRateHisto(h, h5, 1200 * 800)

    print(str((1 - r) * 100) + " de difference")
    execution_time = format(time.time() - a, '.2f')
    print("temps d'execution : " +  str(execution_time))

if __name__ == "__main__":
    main()
