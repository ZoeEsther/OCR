# -*- coding: utf-8 -*-
from PIL import ImageEnhance,Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import feature

# photo_file_path1='D:/OCR/sharpness/sharpness_data/img_25.jpg'
# image1 = Image.open(photo_file_path1)
# enh_sha1 = ImageEnhance.Sharpness(image1)
# image_sharped1 = enh_sha1.enhance(20)

photo_file_path1='D:/OCR/sharpness/sharpness_data/1565.jpg'
image1 = cv2.imread(photo_file_path1)
image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)
canny_1 = feature.canny(img_gray,sigma=1)
canny_2 = feature.canny(img_gray,sigma=2)
canny_3 = feature.canny(img_gray,sigma=3)
# canny_3_RGB = cv2.cvtColor(canny_3,cv2.COLOR_GRAY2BGR)


plt.figure()
plt.subplot(2,2,1)
plt.xticks(())
plt.yticks(())
plt.title("original")
plt.imshow(image1)

plt.subplot(2,2,2)
plt.xticks(())
plt.yticks(())
plt.title("sharpness(sigma=1.0)")
plt.imshow(canny_1)

plt.subplot(2,2,3)
plt.xticks(())
plt.yticks(())
plt.title("sharpness(sigma=2.0)")
plt.imshow(canny_2)

plt.subplot(2,2,4)
plt.xticks(())
plt.yticks(())
plt.title("sharpness(sigma=3.0)")
plt.imshow(canny_3)





plt.show()

