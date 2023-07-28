# import argparse
# import cv2
#
# # 构建参数解析器
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--image", required=True, help="Path to the image")
# # args = vars(ap.parse_args())
#
# # 加载图像
# image = cv2.imread("D:/OCR/picture_processing/images/one_crop.png")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # 构造结构元素（内核），并应用黑帽运算
# rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
# blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
# cv2.imwrite("D:/OCR/picture_processing/images/one_crop_blackhat.png",blackhat)
#
# # 应用白帽运算
# tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
#
# # 输出图像
# cv2.imshow("Original", image)
# cv2.imshow("Blackhat", blackhat)
# # cv2.imshow("Tophat", tophat)
#
# for i in range(0, 3):
# 	dilated = cv2.dilate(blackhat, None, iterations=i + 1)
# 	cv2.imshow("Dilated {} times".format(i + 1), dilated)
#
# cv2.waitKey(0)

#################################  图像锐化 边缘检测算子 ###############
import numpy as np
import matplotlib.pyplot as plt
import random


################################################
#           Robert算子
################################################
def robert_filter(image):
	h = image.shape[0]
	w = image.shape[1]
	image_new = np.zeros(image.shape, np.uint8)
	for i in range(1, h - 1):
		for j in range(1, w - 1):
			image_new[i][j] = np.abs((image[i][j] - image[i + 1][j + 1])) + np.abs(image[i + 1][j] - image[i][j + 1])
	return image_new


################################################
#           Sobel算子
################################################
def sobel_filter(image):
	h = image.shape[0]
	w = image.shape[1]
	image_new = np.zeros(image.shape, np.uint8)

	for i in range(1, h - 1):
		for j in range(1, w - 1):
			sx = (image[i + 1][j - 1] + 2 * image[i + 1][j] + image[i + 1][j + 1]) - \
				 (image[i - 1][j - 1] + 2 * image[i - 1][j] + image[i - 1][j + 1])
			sy = (image[i - 1][j + 1] + 2 * image[i][j + 1] + image[i + 1][j + 1]) - \
				 (image[i - 1][j - 1] + 2 * image[i][j - 1] + image[i + 1][j - 1])
			image_new[i][j] = np.sqrt(np.square(sx) + np.square(sy))
	return image_new


################################################
#           Prewitt算子
################################################
def prewitt_filter(image):
	h = image.shape[0]
	w = image.shape[1]
	image_new = np.zeros(image.shape, np.uint8)

	for i in range(1, h - 1):
		for j in range(1, w - 1):
			sx = (image[i - 1][j - 1] + image[i - 1][j] + image[i - 1][j + 1]) - \
				 (image[i + 1][j - 1] + image[i + 1][j] + image[i + 1][j + 1])
			sy = (image[i - 1][j - 1] + image[i][j - 1] + image[i + 1][j - 1]) - \
				 (image[i - 1][j + 1] + image[i][j + 1] + image[i + 1][j + 1])
			image_new[i][j] = np.sqrt(np.square(sx) + np.square(sy))
	return image_new


################################################
#           Laplacian算子
################################################
def laplacian_filter(image):
	h = image.shape[0]
	w = image.shape[1]
	image_new = np.zeros(image.shape, np.uint8)
	for i in range(1, h - 1):
		for j in range(1, w - 1):
			image_new[i][j] = image[i + 1][j] + image[i - 1][j] + image[i][j + 1] + image[i][j - 1] - 8 * image[i][j]
	return image_new


#############################################################################


if __name__ == "__main__":
	img = plt.imread("1.jpg")

	rgb_weight = [0.299, 0.587, 0.114]
	img_gray = np.dot(img, rgb_weight)

	################################################
	#           原图
	################################################
	plt.subplot(241)
	plt.imshow(img)
	plt.xticks([])
	plt.yticks([])
	plt.title("Original")

	################################################
	#           灰度图
	################################################
	plt.subplot(242)
	plt.imshow(img_gray, cmap=plt.cm.gray)
	plt.xticks([])
	plt.yticks([])
	plt.title("Gray")

	################################################
	#           Robert算子
	################################################
	img_Robert = robert_filter(img_gray)
	img_Robert = img_Robert.astype(np.float64)
	plt.subplot(245)
	plt.imshow(img_Robert, cmap=plt.cm.gray)
	plt.xticks([])
	plt.yticks([])
	plt.title("robert_filter")

	################################################
	#           Sobel算子
	################################################
	img_Sobel = sobel_filter(img_gray)
	img_Sobel = img_Sobel.astype(np.float64)
	plt.subplot(246)
	plt.imshow(img_Sobel, cmap=plt.cm.gray)
	plt.xticks([])
	plt.yticks([])
	plt.title("sobel_filter")

	################################################
	#           Prewitt算子
	################################################
	img_Prewitt = prewitt_filter(img_gray)
	img_Prewitt = img_Prewitt.astype(np.float64)
	plt.subplot(247)
	plt.imshow(img_Prewitt, cmap=plt.cm.gray)
	plt.xticks([])
	plt.yticks([])
	plt.title("prewitt_filter")

	################################################
	#           Laplacian算子
	################################################
	img_Laplacian = laplacian_filter(img_gray)
	img_Laplacian = img_Laplacian.astype(np.float64)
	plt.subplot(248)
	plt.imshow(img_Laplacian, cmap=plt.cm.gray)
	plt.xticks([])
	plt.yticks([])
	plt.title("laplacian_filter")
	plt.show()

