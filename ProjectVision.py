import cv2
import numpy as np
import math
import argparse
from os import walk
import sys
import time

a = time.time()

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
