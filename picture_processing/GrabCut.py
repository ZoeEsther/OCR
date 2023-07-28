import cv2
import numpy as np


  ###############################  切 割  #########################
#边缘检测
def getCanny(image):
    #高斯模糊
    blur=cv2.GaussianBlur(image,(3,3),2,2)
#    cv2.imwrite('2_blur.jpg',blur)
    #边缘检测
    canny=cv2.Canny(blur,60,240,apertureSize=3)
#    cv2.imwrite('3_canny.jpg',canny)
    #膨胀操作，尽量使边缘闭合
    kernel=np.ones((3,3),np.uint8)
    dilate=cv2.dilate(canny,kernel,iterations=1)
#    cv2.imwrite('4_dilate.jpg',dilate)
    return dilate
# 最大轮廓检测
def findMaxContour(image):
    #寻找轮廓
    contours, _ = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #计算面积
    max_contour=[]
    max_area=0.0
    for contour in contours:
        currentArea=cv2.contourArea(contour)
        if currentArea > max_area:
            max_area=currentArea
            max_contour=contour
    max_contour_img=img.copy()
    cv2.drawContours(max_contour_img,max_contour,-1,(0,0,255),3)
#    cv2.imwrite('5_max_contour.jpg',max_contour_img)
    return  max_contour

#四边形顶点检测
def getBoxPoint(contour):
    #多边形拟合凸包
    hull=cv2.convexHull(contour)
    epsilon=0.02*cv2.arcLength(contour,True)
    boxes=cv2.approxPolyDP(hull,epsilon,True)
    boxes=boxes.reshape((len(boxes),2))
    boxes_img=img.copy()
    for box in boxes:
        cv2.circle(boxes_img,tuple(box),5,(0,0,255),2)
#    cv2.imwrite('6_boxes.jpg',boxes_img)
    return boxes

#四边形顶点排序
def orderPoints(boxes):
    rect=np.zeros((4,2),dtype="float32")
    s=boxes.sum(axis=1)
    rect[0] = boxes[np.argmin(s)]
    rect[2] = boxes[np.argmax(s)]
    diff = np.diff(boxes,axis=1)
    rect[1] = boxes[np.argmin(diff)]
    rect[3] = boxes[np.argmax(diff)]
    return rect

#计算长宽
def pointDistance(a,b):
    return  int(np.sqrt(np.sum(np.square(a - b))))

#透视变换
def warpImage(boxes):
    w,h = pointDistance(boxes[0],boxes[1]),pointDistance(boxes[1],boxes[2])
    dst_rect=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],dtype='float32')
    M = cv2.getPerspectiveTransform(boxes,dst_rect)
    warped_img=img.copy()
    warped=cv2.warpPerspective(warped_img,M,(w,h))
    cv2.imwrite('warped.jpg',warped)
    return warped

if __name__=='__main__':
    path='D:/OCR/test_images/tr_img.jpg'
    img=cv2.imread(path)
    dilate=getCanny(img)
    max_contour=findMaxContour(dilate)
    boxes=getBoxPoint(max_contour)
    boxes=orderPoints(boxes)
    warped=warpImage(boxes)