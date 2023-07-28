import tkinter as tk
from tkinter.ttk import Separator
from PIL import Image,ImageTk
import cv2

window = tk.Tk()
window.geometry("1050x580")      # 窗口大小
window.title("字符识别OCR")
window.minsize(180, 1)
window.maxsize(1912, 1038)
window.resizable(1, 1)
window.configure(background="#707070")
window.configure(highlightbackground="#d9d9d9")
window.configure(highlightcolor="black")

##################################### 左边版面用到的函数 #################
def get_local_img(): ############## 获取本地图片
    file =  1

def get_take_photo(): ############### 打开摄像头，获取图片
    cv2.nameWindow("Photo_Detect")
    cap=cv2.VideoCapture(0)
    while (True):
        ret,frame = cap.read()  #视频捕获帧
        cv2.imwrite('D:/OCR_Photo/cap_photo.jpg',frame)
        cv2.imshow('Photo_Detect',frame)
        # 按S 确认图片，传入GUI中显示
        if(cv2.waitKey(1) & 0xFF) == ord('S'): #不断刷新图像，这里是1ms,按 S 拍照
            photo=cv2.resize(frame,(640,480))  #规定图像大小
            cv2.imshow('cap_photo',photo)
            cv2.imwrite('D:/OCR_Photo/cap_photo.jpg',photo)

        if cv2.waitKey(1) & 0xFF == ord('Q'): # 按 Q 关闭所有窗口
            break
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def binaryzation_img():
    ##### 图片灰度二值化函数
    a=1
def binaryzation_value(bv):
    #### 把scale返回的值赋给二值化的阈值，进而调整二值化参数
    bv=0
def sharp_img():
    ##### 图片锐化函数
    b=1
def sharp_value(sv):
    #### 把锐化值赋给锐化函数
    sv=0
def crop_img():
    ##### 图片裁剪
    c=1

def donoising_img():
    ##### 图像去噪
    d=1

def save_pre_img():
   #### 预处理后的图片保存
    e=1

def det_rec_img():
    #### 开始识别
    f=1

def save_text_img():
    ####保存识别结果
    g=1
################# 左边版面(图像选取)  #############
# LabelFrame
img_select = tk.LabelFrame(window,text='图片选取及处理',background="#d9d9d9",relief='groove',foreground="black",
                           highlightbackground="#d9d9d9",highlightcolor="black")
img_select.place(relx=0.0, rely=0.0, relheight=1.002, relwidth=0.300)

# 标签 图片路径
img_path = tk.Label(img_select,text=': 图片路径:',background="#d9d9d9",disabledforeground="#a3a3a3",
                    activebackground="#f9f9f9",activeforeground="black",foreground="#000000",
                    highlightbackground="#d9d9d9",highlightcolor="black")
img_path.place(relx=0.030, rely=0.03, height=30, width=52, bordermode='ignore')

#文本 路径显示
Text1 = tk.Text(img_select,background="white",font="TkTextFont",foreground="black",highlightbackground="#d9d9d9",
                highlightcolor="black",insertbackground="black",wrap="word",selectbackground="blue",
                selectforeground="white")
Text1.place(relx=0.252, rely=0.03, relheight=0.045, relwidth=0.624, bordermode='ignore')

# 按钮 本地图片
local_img = tk.Button(img_select,text='本地图片',background="#d9d9d9",disabledforeground="#a3a3a3",
                      foreground="#000000",highlightbackground="#d9d9d9",highlightcolor="black",pady="0",
                      activeforeground="#000000",activebackground="#ececec",command = get_local_img)
local_img.place(relx=0.250, rely=0.100, height=35, width=135 , bordermode='ignore')

# 按钮 拍照
take_photo = tk.Button(img_select,text='拍照识别',activebackground="#ececec",activeforeground="#000000",
                       background="#d9d9d9",disabledforeground="#a3a3a3",foreground="#000000",
                       highlightbackground="#d9d9d9",highlightcolor="black",pady="0",command = get_take_photo)
take_photo.place(relx=0.250, rely=0.180, height=35, width=135 , bordermode='ignore')

# 2号分割线
TSeparator1 = Separator(img_select)
TSeparator1.place(relx=0.017, rely=0.250, relwidth=0.983, bordermode='ignore')

# 图像灰度二值化 Button
binaryzation = tk.Button(img_select,text='二值化',background="#d9d9d9",disabledforeground="#a3a3a3",
                         foreground="#000000",highlightbackground="#d9d9d9",activeforeground="#000000",
                         activebackground="#ececec",highlightcolor="black",pady="0",
                         command=binaryzation_img)
binaryzation.place(relx=0.020, rely=0.300, height=30, width=75, bordermode='ignore')

# 图像灰度二值化阈值选择 Scale
binaryzation_scale = tk.Scale(img_select, from_=100.0, to=200.0,orient="horizontal",length='300',
                              resolution="5.0",digits="200",tickinterval=20,showvalue=1,
                              command = binaryzation_value)
                              #,activebackground="#ececec",foreground="#000000",highlightbackground="#d9d9d9",
                              # highlightcolor="black",background="#d9d9d9", troughcolor="#808080")
binaryzation_scale.place(relx=0.300, rely=0.250, height=60, width=200)

# 图像锐化处理 button
sharp = tk.Button(img_select,text='锐化',background="#d9d9d9",disabledforeground="#a3a3a3",foreground="#000000",
                  highlightbackground="#d9d9d9",highlightcolor="black",pady="0",activeforeground="#000000",
                  activebackground="#ececec",command = sharp_img)
sharp.place(relx=0.020, rely=0.400, height=30, width=75 , bordermode='ignore')

# 图像锐化值选择 scale
sharp_scale = tk.Scale(img_select, from_=0.0, to=10.0,digits="20",length="30" ,orient="horizontal",
                       resolution="1.0", tickinterval=1,showvalue=1,command = sharp_value)
sharp_scale.place(relx=0.300, rely=0.350, height=60, width=200)

# 图像裁剪 button
crop = tk.Button(img_select,text='裁剪',background="#d9d9d9",highlightcolor="black",disabledforeground="#a3a3a3",
                 foreground="#000000",highlightbackground="#d9d9d9",activeforeground="#000000",
                 activebackground="#ececec",pady="0",
                 command = crop_img)
crop.place(relx=0.020, rely=0.500, height=30, width=75, bordermode='ignore')

# 图像去噪 button
donoising  = tk.Button(img_select,text='去噪',background="#d9d9d9",highlightcolor="black",disabledforeground="#a3a3a3",
                 foreground="#000000",highlightbackground="#d9d9d9",activeforeground="#000000",
                 activebackground="#ececec",pady="0",
                 command = donoising_img)
donoising.place(relx=0.020, rely=0.600, height=30, width=75, bordermode='ignore')

# 预处理后的图片保存 button
save_pre  = tk.Button(img_select,text='预处理图片保存',background="#d9d9d9",highlightcolor="black",
                 disabledforeground="#a3a3a3", activebackground="#ececec",pady="0",
                 foreground="#000000",highlightbackground="#d9d9d9",activeforeground="#000000",
                 command = save_pre_img)
save_pre.place(relx=0.500, rely=0.600, height=30, width=120, bordermode='ignore')

# 分割线2
TSeparator2 = Separator(img_select)
TSeparator2.place(relx=0.017, rely=0.700, relwidth=0.983, bordermode='ignore')

# 开始识别 button
detect_recongnize  = tk.Button(img_select,text='开始识别',background="#d9d9d9",highlightcolor="black",
                 disabledforeground="#a3a3a3", activebackground="#ececec",pady="0",
                 foreground="#000000",highlightbackground="#d9d9d9",activeforeground="#000000",
                 command = det_rec_img)
detect_recongnize.place(relx=0.250, rely=0.750, height=35, width=135, bordermode='ignore')

# 文字保存 button
save_text  = tk.Button(img_select,text='文本保存',background="#d9d9d9",highlightcolor="black",
                 disabledforeground="#a3a3a3", activebackground="#ececec",pady="0",
                 foreground="#000000",highlightbackground="#d9d9d9",activeforeground="#000000",
                 command = save_text_img)
save_text.place(relx=0.250, rely=0.850, height=35, width=135, bordermode='ignore')

##################### 右上版面区域 用到的函数 ######################


###################### 右上版面（图像显示）#####################
img_show = tk.LabelFrame(window,text='图片显示区',background="#d9d9d9",highlightbackground="#d9d9d9",
                         highlightcolor="black",foreground="black",relief='groove')
img_show.place(relx=0.300, rely=0.0, relheight=0.489, relwidth=0.700)

# 原图画布
canvas_original= tk.Canvas(img_show,bg='#eeeeee')
canvas_original.place(relx=0.037,rely=0.027,height=220,width=300)
original_img = tk.Label(img_show,text='原图',background="#d9d9d9")
original_img.place(relx=0.200, rely=0.870, height=30, width=50, bordermode='ignore')

image_file=tk.PhotoImage(file='D:/OCR/test_images/english_test_img/img_1.gif')  ########## 图片路径，暂时代替 只能是gif
image = canvas_original.create_image(0,0,anchor='nw',image=image_file)

# 预处理后的画布
canvas_pre= tk.Canvas(img_show,bg='#eeeeee')
canvas_pre.place(relx=0.520,rely=0.027,height=220,width=300)
pre_img = tk.Label(img_show,text='预处理后的图像',background="#d9d9d9")
pre_img.place(relx=0.650, rely=0.870, height=30, width=100, bordermode='ignore')

############################  右下版面所用函数 ####################

############################  右下版面布置 ####################

# labelFrame
img_result = tk.LabelFrame(window,text='识别结果区',background="#d9d9d9",highlightbackground="#d9d9d9",
                           highlightcolor="black",foreground="black",relief='groove')
img_result.place(relx=0.300, rely=0.482, relheight=0.52, relwidth=0.774)

# 文字检测画布
canvas_result= tk.Canvas(img_result,bg='#eeeeee')
canvas_result.place(relx=0.037,rely=0.027,height=220,width=300)
result_img = tk.Label(img_result,text='文字检测结果',background="#d9d9d9")
result_img.place(relx=0.160, rely=0.870, height=30, width=100, bordermode='ignore')

# 文字展示区
text_result = tk.Text(img_result,background="white",font="TkTextFont",foreground="black",
                                 highlightbackground="#d9d9d9",highlightcolor="black",insertbackground="black",
                                 insertborderwidth="3",selectbackground="blue",selectforeground="white",wrap="none")
text_result.place(relx=0.470, rely=0.090, height=220, width=300, bordermode='ignore')
result_txt = tk.Label(img_result,text='文字识别结果',background="#d9d9d9")
result_txt.place(relx=0.560, rely=0.870, height=30, width=100, bordermode='ignore')

window.mainloop()

