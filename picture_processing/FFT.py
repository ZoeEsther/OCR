# import cv2
# import numpy as np
#
#
# # 二值形态学运算
# def morphology(img):
#     kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 14))  # 腐蚀矩阵
#     iFushi = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel1)  # 对文字腐蚀运算
#     #cv2.imshow('fushi', iFushi)
#
#     kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))  # 膨胀矩阵
#     iPengzhang = cv2.morphologyEx(iFushi, cv2.MORPH_ERODE, kernel2)  # 对背景进行膨胀运算
#    # cv2.imshow('pengzhang', iPengzhang)
#
#     # 背景图和二分图相减-->得到文字
#     jian = np.abs(iPengzhang - img)
#     cv2.imshow("jian", jian)
#
#     kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))  # 膨胀
#     iWenzi = cv2.morphologyEx(jian, cv2.MORPH_DILATE, kernel3)  # 对文字进行膨胀运算
#     cv2.imshow('wenzi', iWenzi)
#
#
# img = cv2.imread("D:/OCR/picture_processing/images/one_crop.png")
# # 1、消除椒盐噪声：
# # 中值滤波器
# median = cv2.medianBlur(img, 3)
# # 消除噪声图
# cv2.imshow("median-image", median)
# # 转化为灰度图
# Grayimg = cv2.cvtColor(median, cv2.COLOR_RGB2GRAY)
# # 2、直方图均衡化：
# hist = cv2.equalizeHist(Grayimg)
# cv2.imshow('hist', hist)
# # 3、二值化处理：
# # 阈值为140
# ret, binary = cv2.threshold(hist, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow("binary-image", binary)
# # 二值形态处理
# morphology(binary)
#
# cv2.waitKey(0)
#
#

######################  以上是腐蚀膨胀 ，无用 #####################

######################  提取颜色 #####################

# import cv2
# import numpy as np
# src = cv2.imread("D:/OCR/picture_processing/images/one_crop.png")
# cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("input", src)
# """
# 提取图中的红色部分
# """
# hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
# cv2.imshow("hsv",hsv)
# low_hsv = np.array([100,43,46])
# high_hsv = np.array([124,255,255])
# mask = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
# # cv2.imshow("test",mask)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# mask_contrary = mask.copy()
# mask_contrary[mask_contrary==0] =1
# mask_contrary[mask_contrary==255]=0
# mask_bool = mask_contrary.astype(bool)
# mask_img=cv2.add(src,np.zeros(np.shape(src),dtype=np.uint8),mask = mask)
# mask_img = cv2.cvtColor(mask_img,cv2.COLOR_BGR2BGRA)
# mask_img[mask_bool]=[0,0,0,0]
#
# cv2.imshow("image",mask_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#############################  RGB 三个通道的显示和组合
# from PIL import Image
# import matplotlib.pyplot as plt        # 可以理解为画板
# import numpy as np
# img = Image.open("D:/OCR/picture_processing/images/one_crop.png")
#
# r,g,b = img.split()   # 分离成RGB三个通道。。提取R G B分量
#
# pic = Image.merge('RGB',(r,g,b))  # 合并通道
#
# plt.figure("beauty")
# plt.subplot(2,3,1), plt.title('R_G_B')  # (x,c,v) 三个数字是可以调节的：  x:表示行   c:表示一行的列数   v:表示第几个
# plt.imshow(img),plt.axis('on')            # 原图 plt.axis是否显示坐标轴
#
# G_B_R = Image.merge('RGB',(g,b,r))  # 合并通道
# plt.subplot(2,3,2), plt.title('G_B_R')   # gray 灰色
# plt.imshow(G_B_R,cmap='gray'),plt.axis('off')
#
# G_R_B = Image.merge('RGB',(g,r,b))  # 合并通道
# plt.subplot(2,3,3), plt.title('G_R_B')   # merge 合并
# plt.imshow(G_R_B),plt.axis('off')
#
# R_B_G = Image.merge('RGB',(r,b,g))  # 合并通道
# plt.subplot(2,3,4), plt.title('R_B_G')   # merge 合并
# plt.imshow(R_B_G),plt.axis('off')
#
# B_G_R = Image.merge('RGB',(b,g,r))  # 合并通道
# plt.subplot(2,3,5), plt.title('B_G_R')   # merge 合并
# plt.imshow(B_G_R),plt.axis('off')
#
# B_R_G = Image.merge('RGB',(b,r,g))  # 合并通道
# plt.subplot(2,3,6), plt.title('B_R_G')   # merge 合并
# plt.imshow(B_R_G),plt.axis('off')
#
# plt.show()

################## H S V
import cv2
import numpy as np

image = cv2.imread('D:/OCR/picture_processing/images/one_crop.png')
HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
H, S, V = cv2.split(HSV)
cv2.imshow('H', H)
cv2.imshow('S', S)
cv2.imshow('V', V)

# cv2.imshow('H-S', H-S)
# cv2.imshow('S-H', S-H)
# cv2.imshow('H-V', H-V)  ## 字白 底黑
# cv2.imshow('V-H',V-H)
#
# cv2.imshow('S-V', S-V)
# cv2.imshow('V-S', V-S)
cv2.imwrite('D:/OCR/picture_processing/images/H.png',H)
cv2.imwrite('D:/OCR/picture_processing/images/S.png',S)
cv2.imwrite('D:/OCR/picture_processing/images/V.png',V)


cv2.waitKey(0)
cv2.destroyAllWindows()

# ################### 巴特沃斯去噪 ###########
# import numpy as np
# import matplotlib as plt
# from PIL import Image
#
# def butterworthPassFilter(image, d, n):
#     f = np.fft.fft2(image)
#     fshift = np.fft.fftshift(f)
#
#     def make_transform_matrix(d):
#         transfor_matrix = np.zeros(image.shape)
#         center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
#         for i in range(transfor_matrix.shape[0]):
#             for j in range(transfor_matrix.shape[1]):
#                 def cal_distance(pa, pb):
#                     from math import sqrt
#                     dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
#                     return dis
#
#                 dis = cal_distance(center_point, (i, j))
#                 transfor_matrix[i, j] = 1 / ((1 + (d / dis)) ** n)
#         return transfor_matrix
#
#     d_matrix = make_transform_matrix(d)
#     new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
#     return new_img
#
#
# img = Image.open("D:/OCR/picture_processing/images/one_crop.png")
#
# plt.subplot(231)
# butter_100_1 = butterworthPassFilter(img,100,1)
# plt.imshow(butter_100_1,cmap="gray")
# plt.title("d=100,n=1")
# plt.axis("off")
# plt.subplot(232)
# butter_100_2 = butterworthPassFilter(img,100,2)
# plt.imshow(butter_100_2,cmap="gray")
# plt.title("d=100,n=2")
# plt.axis("off")
# plt.subplot(233)
# butter_100_3 = butterworthPassFilter(img,100,3)
# plt.imshow(butter_100_3,cmap="gray")
# plt.title("d=100,n=3")
# plt.axis("off")
# plt.subplot(234)
# butter_100_1 = butterworthPassFilter(img,30,1)
# plt.imshow(butter_100_1,cmap="gray")
# plt.title("d=30,n=1")
# plt.axis("off")
# plt.subplot(235)
# butter_100_2 = butterworthPassFilter(img,30,2)
# plt.imshow(butter_100_2,cmap="gray")
# plt.title("d=30,n=2")
# plt.axis("off")
# plt.subplot(236)
# butter_100_3 = butterworthPassFilter(img,30,3)
# plt.imshow(butter_100_3,cmap="gray")
# plt.title("d=30,n=3")
# plt.axis("off")
# plt.show()