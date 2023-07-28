import os
import cv2
from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
####################### 图片增强 ####################

img= None
# 原始图像+ 亮度提升 + 对比度增大 + 锐化
def ImageAugument(path):
        global img
        image =Image.open(path)

        enh_bri = ImageEnhance.Brightness(image) # 亮度增强
        brightness = 1.5
        image_brightened = enh_bri.enhance(brightness)

        enh_con = ImageEnhance.Contrast(image_brightened)#   亮度增强  +  对比度增强
        contrast = 1.5
        image_contrasted = enh_con.enhance(contrast)

        enh_sha = ImageEnhance.Sharpness(image)###  亮度增强 + 对比度增强 + 锐化
        sharpness = 5.0
        image_sharped = enh_sha.enhance(sharpness)
        image_sharped.save("D:/OCR/picture_processing/images/1565_sharp5.jpg")


# 二值形态学运算
def morphology(image):

    Grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #转为灰度图
    ret, binary = cv2.threshold(Grayimg, 150, 255, cv2.THRESH_TOZERO) # 二值化
    cv2.imshow("binary-image", binary)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (20,14))  # 腐蚀矩阵
    iFushi = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel1)  # 对文字腐蚀运算
    #cv2.imshow('fushi', iFushi)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))  # 膨胀矩阵
    iPengzhang = cv2.morphologyEx(iFushi, cv2.MORPH_ERODE, kernel2)  # 对背景进行膨胀运算
    #cv2.imshow('pengzhang', iPengzhang)

    # 背景图和二分图相减-->得到文字
    jian = np.abs(iPengzhang - binary)
    #cv2.imshow("jian", jian)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))  # 膨胀
    iWenzi = cv2.morphologyEx(jian, cv2.MORPH_DILATE, kernel3)  # 对文字进行膨胀运算
    cv2.imwrite("D:/OCR/picture_processing/images/one_crop-light_contrast_sharp.jpg",iWenzi)
    cv2.imshow('wenzi', iWenzi)

    cv2.waitKey(0)


if __name__ == '__main__':
    photo_file_path = "D:/OCR/picture_processing/images/1565.jpg"
    ImageAugument(photo_file_path)

    # img = cv2.imread("D:/OCR/picture_processing/images/one_crop-light_contrast_sharp.jpg")
    # morphology(img)

