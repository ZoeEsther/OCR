import cv2
import numpy as np

def addSaltNoise(img,snr):
    # 指定信噪比
    SNR = snr
    # 获取总共像素个数
    size = img.size
    # 因为信噪比是 SNR ，所以噪声占据百分之10，所以需要对这百分之10加噪声
    noiseSize = int(size * (1 - SNR))
    # 对这些点加噪声
    for k in range(0, noiseSize):
        # 随机获取 某个点
        xi = int(np.random.uniform(0, img.shape[1]))
        xj = int(np.random.uniform(0, img.shape[0]))
        # 增加噪声
        if img.ndim == 2:
            img[xj, xi] = 255
        elif img.ndim == 3:
            img[xj, xi] = 0
    return img

def removeSaltNoise():
    filename = "D:/OCR/Median/median_data/1565.jpg"
    # 得到加噪声之后的图像
    img = addSaltNoise(cv2.imread(filename),0.9)
    # 进行中值滤波
    lbimg = cv2.medianBlur(img, 3)
    cv2.imshow('src', img)
    cv2.imshow('dst', lbimg)
    cv2.waitKey(0)

def main():
    removeSaltNoise()

if __name__ == '__main__':
    main()