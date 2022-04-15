import cv2
import numpy as np
import math
import argparse
from os import walk
from os import path
import sys
import time

a = time.time()


def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), j, count))
        file.flush()

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    file.write("\n")
    file.flush()


for i in progressbar(range(100), "Computing : ", 40):
    time.sleep(0.01)
    execution_time = format(time.time() - a, '.2f')
    # print(execution_time)


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
                        logoAlreadyCaptured.append(box)
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


Images = list()
Images_DataSet = list()
name = list()
Logos = "Logo/Clair/"
DataSet = "Logo/Sur photos/"
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

Draw(Images, name, Images_DataSet)
