import os
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Separator
import tkinter.font as tf
import cv2
import numpy as np
from PIL import Image,ImageTk
from ocr import ocr
import time
import matplotlib.pyplot as plt
from PIL import ImageEnhance
from HSV.HSV_retinex import HSV_autoMSRretinex
from Median.median import adapt_median

img_o = None ##### 原图区域显示的
img_p = None ##### 预处理区域显示的
img_d = None ##### 文字检测区域显示的
img_real_d = None ###真正用来文字识别的
photo_file_path = None  #### 原图路径
pre_img_photo = None   ##### 预处理后的图片路径
again_path = None  ####取消二值化 或 锐化 时，canvas_pre 恢复原图
namei=0

save_pre_photo_folder = 'E:/OCR_Photo_pre'  # 预处理后的图片保存路径文件夹
save_photo_file_path = 'E:/OCR_Photo_Result'   # 文字识别后的图片和文本的保存路径


binaryzation_v = 100
sharp_v = 0
rotate_angle = 0

                     ### 裁剪图片用到的
left_mouse_down_x = 0
left_mouse_down_y = 0
left_mouse_up_x = 0
left_mouse_up_y = 0
sole_rectangle = None
img_1 = None  #### 点击裁剪按钮出现的画布上的图片
i_crop = 0 # 是否有裁剪红色框图 =0，没有

class Application(Frame):
    def __init__(self,master=NONE):
        super().__init__(master)
        self.master= master
        self.place()
        self.creatWidget()



    def donoising_img(self):
        ##### 图像去噪
        d = 1



    '''
     右上版面用到的函数

    '''

    '''
     右下版面用到的函数

    '''



    def creatWidget(self):
        """ 版面一  """
        img_select = LabelFrame(window,text='图片选取及处理',background="#d9d9d9",relief='groove',foreground="black",
                           highlightbackground="#d9d9d9",highlightcolor="black")
        img_select.place(relx=0.0, rely=0.0, relheight=1.002, relwidth=0.300)

        # label ---- 图片路径
        self.img_path = Label(img_select, text=': 图片路径:', background="#d9d9d9", disabledforeground="#a3a3a3",
                            activebackground="#f9f9f9", activeforeground="black", foreground="#000000",
                            highlightbackground="#d9d9d9", highlightcolor="black")
        self.img_path.place(relx=0.030, rely=0.045, height=30, width=52, bordermode='ignore')

        # text ----- 路径显示
        self.Text1 = Label(img_select, background="white", disabledforeground="#a3a3a3",
                            activebackground="#f9f9f9", activeforeground="black", foreground="#000000",
                            highlightbackground="#d9d9d9", highlightcolor="black")
        self.Text1.place(relx=0.252, rely=0.045, relheight=0.045, relwidth=0.624, bordermode='ignore')

        # button ---- 本地图片  链接函数 get_local_img
        self.local_img = Button(img_select, text='本地图片', background="#d9d9d9", disabledforeground="#a3a3a3",
                              foreground="#000000", highlightbackground="#d9d9d9", highlightcolor="black", pady="0",
                              activeforeground="#000000", activebackground="#ececec")
        self.local_img["command"]=get_local_img
        self.local_img.place(relx=0.060, rely=0.120, height=35, width=100, bordermode='ignore')

        # button ----  拍照识别  链接函数 get_take_photo
        self.take_photo = Button(img_select, text='拍照识别', activebackground="#ececec", activeforeground="#000000",
                               background="#d9d9d9", disabledforeground="#a3a3a3", foreground="#000000",
                               highlightbackground="#d9d9d9", highlightcolor="black", pady="0")
        self.take_photo["command"]=get_take_photo
        self.take_photo.place(relx=0.060, rely=0.200, height=35, width=100, bordermode='ignore')

        # separator ---- 分割线
        self.TSeparator1 = Separator(img_select)
        self.TSeparator1.place(relx=0.017, rely=0.280, relwidth=0.983, bordermode='ignore')

        # #  Button ------ 图像灰度二值化  链接函数 binaryzation_img
        # self.binaryzation = Button(img_select, text='二值化', background="#d9d9d9", disabledforeground="#a3a3a3",
        #                          foreground="#000000", highlightbackground="#d9d9d9", activeforeground="#000000",
        #                          activebackground="#ececec", highlightcolor="black", pady="0")
        # self.binaryzation["command"] = binaryzation_img
        # self.binaryzation.place(relx=0.020, rely=0.900, height=30, width=75, bordermode='ignore')
        #
        # #  Scale ----- 图像灰度二值化阈值选择 返回阈值 binaryzation_value
        # self.binaryzation_scale = Scale(img_select, from_=100.0, to=200.0, orient="horizontal", length='300',
        #                               resolution="2.0", digits="200", tickinterval=20, showvalue=1,
        #                               )
        # self.binaryzation_scale["command"] = binaryzation_value
        # self.binaryzation_scale.place(relx=0.300, rely=0.850, height=60, width=200)

        # button ----- 图像锐化处理 链接函数 sharp_img
        self.sharp = Button(img_select, text='锐化', background="#d9d9d9", disabledforeground="#a3a3a3",
                          foreground="#000000",
                          highlightbackground="#d9d9d9", highlightcolor="black", pady="0", activeforeground="#000000",
                          activebackground="#ececec" )
        self.sharp["command"] = sharp_img
        self.sharp.place(relx=0.020, rely=0.330, height=30, width=75, bordermode='ignore')

        # scale ----- 图像锐化值选择 返回锐化值 sharp_value
        self.sharp_scale = Scale(img_select, from_=0.0, to=20.0, digits="20", length="30", orient="horizontal",
                               resolution="2.0", tickinterval=1, showvalue=1)
        self.sharp_scale["command"] = sharp_value
        self.sharp_scale.place(relx=0.300, rely=0.280, height=60, width=200)

        # button ----  图像裁剪  链接函数 crop_img
        self.crop =Button(img_select, text='裁剪', background="#d9d9d9", highlightcolor="black",
                         disabledforeground="#a3a3a3",
                         foreground="#000000", highlightbackground="#d9d9d9", activeforeground="#000000",
                         activebackground="#ececec", pady="0",
                         )
        self.crop["command"] = crop_img
        self.crop.place(relx=0.350, rely=0.790, height=30, width=75, bordermode='ignore')

        # button ----  HSV空间Retinex色彩增强  链接函数 HSV_retinex函数
        self.crop =Button(img_select, text='色彩增强', background="#d9d9d9", highlightcolor="black",
                         disabledforeground="#a3a3a3",
                         foreground="#000000", highlightbackground="#d9d9d9", activeforeground="#000000",
                         activebackground="#ececec", pady="0",
                         )
        self.crop["command"] = HSV_retinex
        self.crop.place(relx=0.350, rely=0.630, height=30, width=75, bordermode='ignore')

        # button ----  中值滤波  链接函数 adapt_medianBlur函数
        self.crop =Button(img_select, text='中值滤波', background="#d9d9d9", highlightcolor="black",
                         disabledforeground="#a3a3a3",
                         foreground="#000000", highlightbackground="#d9d9d9", activeforeground="#000000",
                         activebackground="#ececec", pady="0",
                         )
        self.crop["command"] = adapt_medianBlur
        self.crop.place(relx=0.350, rely=0.710, height=30, width=75, bordermode='ignore')

        # button ----  高斯滤波  链接函数 adapt_median函数  GaussianBlur
        self.crop =Button(img_select, text='高斯滤波', background="#d9d9d9", highlightcolor="black",
                         disabledforeground="#a3a3a3",
                         foreground="#000000", highlightbackground="#d9d9d9", activeforeground="#000000",
                         activebackground="#ececec", pady="0",
                         )
        self.crop["command"] = GaussianBlur
        self.crop.place(relx=0.350, rely=0.550, height=30, width=75, bordermode='ignore')

        # button ----  图像旋转  链接函数 rotate_img
        self.rotate =Button(img_select, text='旋转', background="#d9d9d9", highlightcolor="black",
                         disabledforeground="#a3a3a3",
                         foreground="#000000", highlightbackground="#d9d9d9", activeforeground="#000000",
                         activebackground="#ececec", pady="0",
                         )
        self.rotate["command"] = rotate_img
        self.rotate.place(relx=0.020, rely=0.430, height=30, width=75, bordermode='ignore')

        # scale ----- 图像旋转角度 返回锐化值 rotate_angle
        self.rotate_angle = Scale(img_select, from_=-15, to=15.0, digits="20", length="30", orient="horizontal",
                               resolution="1.0", tickinterval=5, showvalue=1)
        self.rotate_angle["command"] = rotate_value
        self.rotate_angle.place(relx=0.300, rely=0.380, height=60, width=200)

        #  button ----  预处理后的图片保存  链接函数 save_pre_img
        self.recover = Button(img_select, text='恢复至原图', background="#d9d9d9", highlightcolor="black",
                             disabledforeground="#a3a3a3", activebackground="#ececec", pady="0",
                             foreground="#000000", highlightbackground="#d9d9d9", activeforeground="#000000",
                             )
        self.recover["command"]= recover
        self.recover.place(relx=0.280, rely=0.870, height=30, width=120, bordermode='ignore')

        # 分割线2
        self.TSeparator2 = Separator(img_select)
        self.TSeparator2.place(relx=0.017, rely=0.520, relwidth=0.983, bordermode='ignore')

        #  button ----  开始识别 链接函数 det_rec_img
        self.detect_recongnize = Button(img_select, text='开始识别', background="#d9d9d9", highlightcolor="black",
                                      disabledforeground="#a3a3a3", activebackground="#ececec", pady="0",
                                      foreground="#000000", highlightbackground="#d9d9d9", activeforeground="#000000",
                                      )
        self.detect_recongnize["command"]=det_rec_img
        self.detect_recongnize.place(relx=0.550, rely=0.120, height=35, width=100, bordermode='ignore')

        # button ----  文字保存  链接函数 save_text_img
        self.save_text = Button(img_select, text='文本保存', background="#d9d9d9", highlightcolor="black",
                              disabledforeground="#a3a3a3", activebackground="#ececec", pady="0",
                              foreground="#000000", highlightbackground="#d9d9d9", activeforeground="#000000",
                              )
        self.save_text["command"]=save_text_img
        self.save_text.place(relx=0.550, rely=0.200, height=35, width=100, bordermode='ignore')

        ###################### 右上版面（图像显示）#####################
        img_show = LabelFrame(window, text='图片显示区', background="#d9d9d9", highlightbackground="#d9d9d9",
                                 highlightcolor="black", foreground="black", relief='groove')
        img_show.place(relx=0.300, rely=0.0, relheight=0.489, relwidth=0.700)

        # canvas 放原图
        self.canvas_original = Canvas(img_show, bg='#eeeeee')
        self.canvas_original.place(relx=0.037, rely=0.027, height=260, width=380)
        self.original_img = Label(img_show, text='原图', background="#d9d9d9")
        self.original_img.place(relx=0.225, rely=0.890, height=30, width=50, bordermode='ignore')

        # canvas 放处理后的图片的画布
        self.canvas_pre = Canvas(img_show, bg='#eeeeee')
        self.canvas_pre.place(relx=0.520, rely=0.027, height=260, width=380)
        self.pre_img = Label(img_show, text='预处理后的图像', background="#d9d9d9")
        self.pre_img.place(relx=0.675, rely=0.900, height=30, width=100, bordermode='ignore')

        ############################  右下版面布置 ####################
        img_result = LabelFrame(window, text='识别结果区', background="#d9d9d9", highlightbackground="#d9d9d9",
                           highlightcolor="black", foreground="black", relief='groove')
        img_result.place(relx=0.300, rely=0.482, relheight=0.52, relwidth=0.774)

        # canvas 文字检测后的图片的画布
        self.canvas_result = Canvas(img_result, bg='#eeeeee')
        self.canvas_result.place(relx=0.037, rely=0.027, height=260, width=380)
        self.result_img = Label(img_result, text='文字检测结果', background="#d9d9d9")
        self.result_img.place(relx=0.180, rely=0.860, height=30, width=100, bordermode='ignore')

        # text 文字识别的结果展示区
        self.text_result = Text(img_result, background="white", foreground="black",font = ("Times",15),
                              highlightbackground="#d9d9d9", highlightcolor="black", insertbackground="black",
                              insertborderwidth="3", selectbackground="blue", selectforeground="white", wrap="none")

        self.text_result.place(relx=0.470, rely=0.090, height=260, width=380, bordermode='ignore')
        self.result_txt = Label(img_result, text='文字识别结果', background="#d9d9d9")
        self.result_txt.place(relx=0.610, rely=0.860, height=30, width=100, bordermode='ignore')


'''
 左边版面用到的函数

'''

def get_local_img():  ############## 获取本地图片
        app.canvas_original.delete("all")
        app.canvas_pre.delete('all')
        app.canvas_result.delete('all')
        app.text_result.delete(1.0 , END )

        global img_o, img_real_d,photo_file_path,again_path

        file_path = askopenfilename(title="选择图片",initialdir="E:/OCR_Photo",filetypes=[("All Files","*")])
        photo_file_path  = file_path
        again_path = photo_file_path
        app.Text1["text"]=photo_file_path

        img_open = Image.open(file_path)
        img_real_d = img_open

        img_open = img_open.resize((380,260),Image.BICUBIC)
        img_o = ImageTk.PhotoImage(img_open)
        app.canvas_original.create_image(0,0,anchor=NW ,image=img_o)
        app.canvas_pre.create_image(0,0,anchor=NW ,image=img_o)



def cap():
    global photo_file_path,img_o,img_real_d,again_path,namei
    capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 开启摄像头
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用XVID编码器
    # out = cv2.VideoWriter("D:/OCR_Photo/output.avi",fourcc,20.0,(640,480))  #  保存文件名、编码器、帧率、视频宽高
    while True:
         ret,frame = capture.read()
         # frame = cv2.flip(frame,1)  ##  翻转画面
       #  out.write(frame)  # 保存录像结果
         cv2.imshow("frame", frame)
         input = cv2.waitKey(1)  # 等待键盘的输入，1 表示延时1ms 切换到下一帧图像
         if input == ord(' '):   # 键盘输入 q ,停止
             namei = int(namei)
             namei=namei+1
             namei = str(namei)
             name =  os.path.join('E:/OCR_Photo/photo_'+namei+".bmp")
             cv2.imwrite(name, frame)
             frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
             photo_file_path = name
             again_path=photo_file_path
             img_o = frame
             break
    capture.release()
    #out.release()
    cv2.destroyAllWindows()

def get_take_photo():  ############### 打开摄像头，获取图片
    global photo_file_path,img_o
    app.canvas_original.delete("all")
    app.canvas_pre.delete('all')
    app.canvas_result.delete('all')
    app.text_result.delete(1.0 , END)

    cap()
    img_o = Image.fromarray(img_o)
    img_o = img_o.resize((380,260),Image.BICUBIC)
    img_o = ImageTk.PhotoImage(img_o)
    app.canvas_original.create_image(0, 0, anchor=NW, image=img_o)
    app.canvas_pre.create_image(0, 0, anchor=NW, image=img_o)
    app.Text1["text"] = photo_file_path

def binaryzation_value(v):  #### 把scale返回的值赋给二值化的阈值，进而调整二值化参数
    global binaryzation_v
    binaryzation_v = v
    return binaryzation_v
def sharp_value(v):   #### 锐化拉条
    global sharp_v
    sharp_v = v
    return sharp_v
def rotate_value(v): #### 旋转拉条
    global rotate_angle
    rotate_angle = v
    return rotate_angle


def recover():  ### 恢复至原图
    global  again_path,photo_file_path
    photo_file_path = again_path
    app.canvas_pre.delete('all')
    img_show(photo_file_path)

def binaryzation_img():    ##### 图片灰度二值化函数
    global binaryzation_v, photo_file_path,pre_img_photo,save_pre_photo_folder,sharp_v,again_path
    app.canvas_pre.delete('all')

    img = cv2.imread(photo_file_path)
    Grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转为灰度图
    thresh = int(binaryzation_v)
    ret, binary = cv2.threshold(Grayimg, thresh, 255, cv2.THRESH_TOZERO)  # 二值化
    pre_img_photo =  os.path.join(save_pre_photo_folder + '/' + photo_file_path.split('/')[-1].split('.')[0] + ".png")
    cv2.imwrite(pre_img_photo, binary)
    img_show(pre_img_photo)
    photo_file_path = pre_img_photo
   # print("二值化完成，保存在：{}".format(pre_img_photo))


def sharp_img():  ### 图像锐化

    global  photo_file_path, pre_img_photo, save_pre_photo_folder, sharp_v,again_path
    app.canvas_pre.delete('all')
    again_path = photo_file_path

    image = Image.open(photo_file_path)
    # enh_bri = ImageEnhance.Brightness(image)  # 亮度增强
    # brightness = 1.5
    # image_brightened = enh_bri.enhance(brightness)
    # # print("亮度增强完成")
    # enh_con = ImageEnhance.Contrast(image_brightened)  # 亮度增强  +  对比度增强
    # contrast = 1.5
    # image_contrasted = enh_con.enhance(contrast)
    # #  print("亮度增强+对比度增强完成")
    enh_sha = ImageEnhance.Sharpness(image)  ###  亮度增强 + 对比度增强 + 锐化
    sharpness = int(sharp_v)
    image_sharped = enh_sha.enhance(sharpness)
    #  print("亮度增强+对比度增强+ 锐化 完成")
    pre_img_photo = os.path.join(save_pre_photo_folder + '/' + photo_file_path.split('/')[-1].split('.')[0] + ".png")
    image_sharped.save(pre_img_photo)
    img_show(pre_img_photo)
    photo_file_path = pre_img_photo


def single_pic_proc(image_path):
    image = np.array(Image.open(image_path).convert('RGB'))
    result, image_framed = ocr(image)
    return result,image_framed

def get_ocr_img_txt():  # 文字识别
    global  img_d,txt_file,save_photo_file_path
    #  t = time.time()
    result, image_framed = single_pic_proc(photo_file_path)

    output_file = os.path.join(save_photo_file_path, photo_file_path.split('/')[-1].split('.')[0] + ".png")
    txt_file = os.path.join(save_photo_file_path, photo_file_path.split('/')[-1].split('.')[0] + '.txt')
    txt_f = open(txt_file, 'w',encoding='utf-8')
    Image.fromarray(image_framed).save(output_file)  # 文字检测图片保存
    img_d = Image.fromarray(image_framed)

    for key in result:
        #        print(result[key][1])
        txt_f.write(result[key][1] + '\n')

    txt_f.close()


def det_rec_img(): #  开始识别按钮
    global img_d,txt_file

    app.canvas_result.delete("all")
    app.text_result.delete(1.0, END)

    get_ocr_img_txt()
    img_w,img_h = img_d.size

    if img_w != 380 & img_h !=260:
        img_d = img_d.resize((380,260),Image.BICUBIC)

    img_d = ImageTk.PhotoImage(img_d)
    app.canvas_result.create_image(0, 0, anchor=NW, image=img_d)

    f = open(txt_file,'r',encoding='utf-8',errors='ignore')
    for i in f:
        app.text_result.insert(END, str(i))
    f.close()


def save_text_img(): ####保存识别结果
    global txt_file
    text = app.text_result.get("1.0","end")
    f = open(txt_file,'w',encoding = 'utf-8')
    f.write(text+'\n')
    f.close()
    btnClick()

def btnClick():
    messagebox.showinfo("消息","以成功保存至"+"\n"+ txt_file)

def crop_img():   ##### 图片裁剪
    global  photo_file_path,pre_img_photo,save_pre_photo_folder,img_real_d,i_crop,again_path
    global  left_mouse_down_x, left_mouse_down_y, left_mouse_up_x, left_mouse_up_y

    again_path=photo_file_path
    pre_img_photo = os.path.join(save_pre_photo_folder + '/' + photo_file_path.split('/')[-1].split('.')[0] + '.png')

    app.canvas_pre.bind('<Button-1>',left_mouse_down)
    app.canvas_pre.bind('<ButtonRelease-1>', left_mouse_up)
    app.canvas_pre.bind('<B1-Motion>', moving_mouse)

    if i_crop  == 1:
        temp = messagebox.askokcancel(title="裁剪", message="请再次确认裁剪区域！")

        if temp :
            crop_start(photo_file_path, pre_img_photo, left_mouse_down_x, left_mouse_down_y, left_mouse_up_x, left_mouse_up_y)
            app.canvas_pre.delete("all")
            img_show(pre_img_photo)
            photo_file_path = pre_img_photo
            i_crop = 0

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image


    if width is None:

        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def img_show(img_path):
    global img_p

    img_p = Image.open(img_path)
    img_p = img_p.resize((380,260),Image.BICUBIC)
    img_p = ImageTk.PhotoImage(img_p)

    app.canvas_pre.create_image(0, 0, anchor=NW , image=img_p)


def crop_start(source_path, save_path, x_begin, y_begin, x_end, y_end):
    if x_begin < x_end:
        min_x = x_begin
        max_x = x_end
    else:
        min_x = x_end
        max_x = x_begin
    if y_begin < y_end:
        min_y = y_begin
        max_y = y_end
    else:
        min_y = y_end
        max_y = y_begin
    #save_path = os.path.abspath(save_path)
    img = Image.open(source_path)
    img = img.resize((380,260),Image.BICUBIC)
    region = img.crop((min_x, min_y, max_x, max_y))
    region.save(save_path)

    print('裁剪完成,保存于:{}'.format(save_path))



def left_mouse_down(event):  # 鼠标左键按下
    global left_mouse_down_x,left_mouse_down_y
    left_mouse_down_x = event.x
    left_mouse_down_y = event.y

def left_mouse_up(event):  # 鼠标左键释放
    global left_mouse_up_x,left_mouse_up_y
    left_mouse_up_x = event.x
    left_mouse_up_y = event.y

def moving_mouse(event):  # print('鼠标左键按下并移动')
    global sole_rectangle
    global left_mouse_down_x, left_mouse_down_y
    global i_crop
    moving_mouse_x = event.x
    moving_mouse_y = event.y
    if sole_rectangle is not None:
        app.canvas_pre.delete(sole_rectangle) # 删除前一个矩形
        i_crop = 0
    sole_rectangle = app.canvas_pre.create_rectangle(left_mouse_down_x, left_mouse_down_y, moving_mouse_x,
                       moving_mouse_y, outline='red')
    i_crop = 1


def  rotate_img(): ####  图像旋转
    global  photo_file_path,rotate_angle,pre_img_photo
    print(photo_file_path)
    angle = int(rotate_angle)
    image = cv2.imread(photo_file_path)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    img = cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
    pre_img_photo = os.path.join(save_pre_photo_folder + '/' + photo_file_path.split('/')[-1].split('.')[0] + ".png")
    cv2.imwrite(pre_img_photo,img)
    photo_file_path = pre_img_photo
    img_show(pre_img_photo)

def HSV_retinex():
    global photo_file_path,pre_img_photo
    img_auto = HSV_autoMSRretinex(photo_file_path)
    img_auto = Image.fromarray(img_auto)
    pre_img_photo = os.path.join(save_pre_photo_folder + '/' + photo_file_path.split('/')[-1].split('.')[0] + ".png")
    img_auto.save(pre_img_photo)
    img_show(pre_img_photo)
    photo_file_path = pre_img_photo

def adapt_medianBlur():
    global photo_file_path, pre_img_photo
    img_median = adapt_median(photo_file_path)
    img_median = Image.fromarray(img_median)
    pre_img_photo = os.path.join(save_pre_photo_folder + '/' + photo_file_path.split('/')[-1].split('.')[0] + ".png")
    img_median.save(pre_img_photo)
    img_show(pre_img_photo)
    photo_file_path = pre_img_photo

def GaussianBlur():
    global photo_file_path, pre_img_photo
    img_gs = cv2.imread(photo_file_path)
    img_GaussianBlur5 = cv2.GaussianBlur(img_gs, (5,5), 0)
    img_GaussianBlur5 = cv2.cvtColor(img_GaussianBlur5,cv2.COLOR_BGR2RGB)
    img_GaussianBlur5 = Image.fromarray(img_GaussianBlur5)
    pre_img_photo = os.path.join(save_pre_photo_folder + '/' + photo_file_path.split('/')[-1].split('.')[0] + ".png")
    img_GaussianBlur5.save(pre_img_photo)
    img_show(pre_img_photo)
    photo_file_path = pre_img_photo



if __name__ == '__main__':
    window = Tk()
    window.geometry("1050x580")  # 窗口大小
    window.title("字符识别OCR")
    window.minsize(180, 1)
    window.maxsize(1912, 1038)
    window.resizable(1, 1)
    window.configure(background="#707070")
    window.configure(highlightbackground="#d9d9d9")
    window.configure(highlightcolor="black")

    app = Application(master=window)

    window.mainloop()