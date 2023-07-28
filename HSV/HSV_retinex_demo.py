import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import PIL as Image
import json
from HSV.HSV_retinex import HSV_autoMSRretinex,HSV_MSRCRretinex

image_path = "D:/OCR/HSV/HSV_data/test2.png"
src = cv2.imread(image_path)
img_auto = HSV_autoMSRretinex(image_path)

h_all=np.hstack((src,img_auto))
cv2.imshow("HSV_autoMSR",h_all)

cv2.waitKey(0)
cv2.destroyAllWindows()