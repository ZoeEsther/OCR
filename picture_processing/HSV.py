import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

###########################获取图片的灰度图##########################
def getGray():
     p = [] # 列表存放图像
     img = cv2.imread("D:/OCR/picture_processing/images/one_crop.png")
# cv2.imshow读取图片的NumPy arrays数组是以 BGR order形式保存的，
# 而Matplotlib中的plt.imshow 是以RGB顺序保存的。
# 存入列表时进行转换以便用plt.show()显示时为RGB格式图像。
     p.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

# 读入正常图像并进行灰度化处理
     gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) # 转为灰度图像
     p.append(cv2.cvtColor(gray,cv2.COLOR_BGR2RGB))

# 对灰度图像进行二值化处理 设置全局阈值为127
     ret,gray1 = cv2.threshold(gray,160,255,cv2.THRESH_BINARY)
     cv2.imwrite('D:/OCR/test_images/Sharp_gray.jpg',gray1)
     p.append(cv2.cvtColor(gray1,cv2.COLOR_BGR2RGB))

# plt.show():在一个窗口显示多幅图片
     for i in range(3):
 # subplot(行,列,索引)
         plt.subplot(1,3,i+1)
         plt.imshow(p[i])
    # 没有坐标轴刻度
         plt.xticks([])
         plt.yticks([])
     plt.show()

###################################获取图片的彩色直方图################################
def getColor_zft():
    b=[]
    src = cv2.imread('D:\\OCR\\test_images\\******.jpg')
    histb = cv2.calcHist([src], [0], None, [256], [0, 255])
    histg = cv2.calcHist([src], [1], None, [256], [0, 255])
    histr = cv2.calcHist([src], [2], None, [256], [0, 255])

    cv2.imshow("src", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.plot(histb, color='b')
    plt.plot(histg, color='g')
    plt.plot(histr, color='r')
    plt.show()

if __name__ == '__main__':
    getGray()
#    getColor_zft()
# cv2.waitKey(0)
# cv2.destroyAllWindows()