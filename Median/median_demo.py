from Median.median import adapt_median_filter,sp_noise,gasuss_noise,gaussian_noise
import cv2
import matplotlib.pyplot as plt

def adapt_median(img_path):
    image = cv2.imread(img_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_adapt_median = adapt_median_filter(image_gray, 3, 7)
    return image_adapt_median

def makesp_noise(img_path):
    image = cv2.imread(img_path)
    out1 = sp_noise(image, prob=0.06)
    return out1

def makegs_noise(img_path):
    image = cv2.imread(img_path)
    out2 = gasuss_noise(image, mean=0, var=0.008)
    return out2

def add_gs(img_path):
    image = cv2.imread(img_path)
    img_gs= gaussian_noise(image,mean=0,sigma=0.1)

    return img_gs


if __name__ == '__main__':

    # img_path = "D:/OCR/Median/median_data/1565.jpg"
    # photo_sp=makesp_noise(img_path)
    # cv2.imshow("photo_sp",photo_sp)
    # cv2.imwrite("D:/OCR/Median/median_data/1565-sp.jpg",photo_sp)

    # image_gray = cv2.cvtColor(photo_sp, cv2.COLOR_BGR2GRAY)
    # image_adapt_median = adapt_median_filter(image_gray, 3, 7)
    # cv2.imshow("image_adapt_median",image_adapt_median)

    # image_general_median_3=cv2.medianBlur(photo_sp,5)
    # image_general_median_3=cv2.cvtColor(image_general_median_3, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("image_general_median_3.png",image_general_median_3)
    # cv2.imshow("image_general_median_3",image_general_median_3)
    #
    # image_general_median_5=cv2.medianBlur(img_1565_sp,5)
    # image_general_median_5=cv2.cvtColor(image_general_median_5, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("image_general_median_5",image_general_median_5)

    # cv2.waitKey(0)

    img_path = "D:/OCR/Median/median_data/-2001-qx.png"
    img_1565_gs = add_gs(img_path)
    # img_1565_gs = cv2.cvtColor(img_1565_gs,cv2.COLOR_BGR2RGB)
    cv2.imshow("-2001-qx", img_1565_gs)
    cv2.imwrite("-2001-qx-gs.png",img_1565_gs)

    img_GaussianBlur3 = cv2.GaussianBlur(img_1565_gs, (3, 3), 0)
    cv2.imshow("img_GaussianBlur3", img_GaussianBlur3)
    cv2.imwrite("-2001-qx-gsBlur3.png",img_GaussianBlur3)

    cv2.waitKey(0)


    # img_GaussianBlur1 = cv2.GaussianBlur(img_1565_gs,(1,1),0)
    # img_GaussianBlur3 = cv2.GaussianBlur(img_1565_gs, (3,3), 0)
    # img_GaussianBlur5 = cv2.GaussianBlur(img_1565_gs, (5,5), 0)
    # img_GaussianBlur7 = cv2.GaussianBlur(img_1565_gs, (7,7), 0)
    # # cv2.imwrite("D:/OCR/picture_processing/images/mohu_1565.png",img_GaussianBlur7)
    # #
    # # img_GaussianBlur1 = cv2.cvtColor(img_GaussianBlur1, cv2.COLOR_BGR2RGB)
    # # img_GaussianBlur3 = cv2.cvtColor(img_GaussianBlur3, cv2.COLOR_BGR2RGB)
    # # img_GaussianBlur5 = cv2.cvtColor(img_GaussianBlur5, cv2.COLOR_BGR2RGB)
    # # img_GaussianBlur7 = cv2.cvtColor(img_GaussianBlur7, cv2.COLOR_BGR2RGB)
    #
    # plt.figure()

    # plt.subplot(3,2,1)
    # plt.xticks(())
    # plt.yticks(())
    # plt.title("src")
    # plt.imshow(img_1565_gs)
    #
    # plt.subplot(3,2,3)
    # plt.xticks(())
    # plt.yticks(())
    # plt.title("Ksize=(1,1)")
    # plt.imshow(img_GaussianBlur1)
    #
    # plt.subplot(3,2,5)
    # plt.xticks(())
    # plt.yticks(())
    # plt.title("Ksize=(3,3)")
    # plt.imshow(img_GaussianBlur3)

    # plt.subplot(3,2,1)
    # plt.xticks(())
    # plt.yticks(())
    # plt.title("Ksize=(5,5)")
    # plt.imshow(img_GaussianBlur5)
    #
    # plt.subplot(3,2,3)
    # plt.xticks(())
    # plt.yticks(())
    # plt.title("Ksize=(7,7)")
    # plt.imshow(img_GaussianBlur7)

    # plt.subplot(3,2,2)
    # plt.ylim(0,5000)
    # plt.hist(img_1565_gs.ravel(), bins=256, range=[0, 256])
    #
    # plt.subplot(3,2,4)
    # plt.ylim(0, 5000)
    # plt.hist(img_GaussianBlur1.ravel(), bins=256, range=[0, 256])
    #
    # plt.subplot(3,2,6)
    # plt.ylim(0, 15000)
    # plt.hist(img_GaussianBlur3.ravel(), bins=256, range=[0, 256])
    #
    # plt.subplot(3,2,2)
    # plt.ylim(0, 15000)
    # plt.hist(img_GaussianBlur5.ravel(), bins=256, range=[0, 256])
    #
    # plt.subplot(3,2,4)
    # plt.ylim(0, 15000)
    # plt.hist(img_GaussianBlur7.ravel(), bins=256, range=[0, 256])
    #
    # plt.show()
    #
    #
    #
    #
    #
    #
    #
    #





