import cv2
import numpy as np
import math
import argparse
from os import walk
import sys
import time
import Histo











def main():
    a = time.time()
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