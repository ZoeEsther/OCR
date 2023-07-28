# OCR
# ----------------------------- 中文 -----------------------------
# 本项目实现了基于CRNN和CTPN的产品识别字符的识别，主要涉及英文字母和标点符号。它包括一个可视化的GUI界面，允许自适应调整过度曝光或黑暗的图像。

# 环境版本
    训练环境: PyCharm+Python3.7+GPU
    测试环境: PyCharm+Python3.7+CPU
    主要Packages及对应版本:
	               torch       	            1.8.1
	               torchvision	            0.9.1
	               OpenCv-python           	    4.5.1.48
	               numpy	                    1.16.6
                       cv                           1.0.0
                       matplotlib                   3.3.2
                       pandas                       1.1.4
                       trans                        2.1.0

# 代码功能
     a. 文件夹train_code：包含train_ctpn、train_crnn两个文件夹，前者为文本检测模型CTPN训练文件夹，后者为文字识别模型CRNN的训练文件夹。
         将模型训练需要的训练数据放入对应的train_data（CRNN网络模型还需要验证集val_data)，运行ctpn_train.py、train_pytorch_ctc.py即开始训练模型。
         其他.py文件均内含训练模型所需的函数。
     b. 文件夹checkpoints：包含网络模型的训练结果；
     c. 文件夹detect：文本检测CTPN测试代码；
     d. 文件夹recognize：文字识别CRNN测试代码；
     e. 文件夹GUI：界面代码；
     f. 文件夹HSV、Median、sharpness等其他均为图像处理代码；
     g. demo.py是将文本检测与文字识别结合在一起的有完整字符识别功能的代码；

# 可视化效果图
GUI界面及识别效果图：
![image](https://github.com/ZoeEsther/OCR/assets/119051069/0e3563e7-abfd-4798-aecc-a2aaeb78cb23)

文本检测效果：（ICDAR2013_Incident Scene Text 官方端到端评估平台）
![image](https://github.com/ZoeEsther/OCR/assets/119051069/d6ef03ae-4a1a-4604-ab91-f09553999f88)

界面允许手动的图像预处理，其效果图：
① 旋转 
![image](https://github.com/ZoeEsther/OCR/assets/119051069/6cbcea72-f8d1-4ea6-a997-f3e39486c7be)

② 锐化
![image](https://github.com/ZoeEsther/OCR/assets/119051069/1a294b50-d26f-48d8-b2a8-fbaa81e2d52b)

③ 色彩增强
![image](https://github.com/ZoeEsther/OCR/assets/119051069/e440497a-0fec-4cc2-ba14-e4dd09e244a5)

④ 滤波
![image](https://github.com/ZoeEsther/OCR/assets/119051069/243d1970-2888-428e-8f49-66ef6c7277a9)


# ----------------------------- English -----------------------------
# This project realizes the recognition of product identification cahracters based on CRNN and CTPN, mainly involving English alphabet and Punctuation. It includes a visual GUI interface that allows for adaptive adjustment of overexposed or dark images.

# Code Function
     a. Folder 'train_ Code': The folder contains two sub folders, namely train_ctpn and train_crnn. 
        The former is used to train the text detection model CTPN, and the latter is used to train the text recognition model CRNN.
        Put each training set into the corresponding folder named train_data, and run ctpn_train.py or train_pytoorch_ctc.py to start training the model.
        Other files with .py suffixes are functions or classes needed to train the model.
     b. Folder 'checkpoints': The folder is used to store network training results.
     c. Folder 'detect': The folder contains test code for text detection related to CTPN.
     d. Folder 'recognize': The folder contains test code for text recognition related to CRNN.
     e. The GUI folder is the interface code.
     f. HSV, Median, sharpness and other folders are image processing-related codes.
     g. demo.py is a code that combines text detection and text recognition with complete character recognition function.

# Configuration environment
     Major packages and corresponding versions:
         torch       	                1.8.1
         torchvision	                0.9.1
         OpenCv-python           	    4.5.1.48
         numpy	                      1.16.6
         cv                           1.0.0
         matplotlib                   3.3.2
         pandas                       1.1.4
         trans                        2.1.0
     Training environment: pycharm + python3.7+GPU
     Test enviroment: pycharm + python3.7+CPU


# Visual rendering 
GUI interface and recognition renderings:
![image](https://github.com/ZoeEsther/OCR/assets/119051069/0e3563e7-abfd-4798-aecc-a2aaeb78cb23)

Text detection effect：（"ICDAR2013_Incident Scene Text” official end-to-end evaluation platform）
![image](https://github.com/ZoeEsther/OCR/assets/119051069/d6ef03ae-4a1a-4604-ab91-f09553999f88)

The interface allows for manual image preprocessing, and its rendering are:
① Rotation
![image](https://github.com/ZoeEsther/OCR/assets/119051069/6cbcea72-f8d1-4ea6-a997-f3e39486c7be)

② Sharpening
![image](https://github.com/ZoeEsther/OCR/assets/119051069/1a294b50-d26f-48d8-b2a8-fbaa81e2d52b)

③ Color enhancement
![image](https://github.com/ZoeEsther/OCR/assets/119051069/e440497a-0fec-4cc2-ba14-e4dd09e244a5)

④ filtering
![image](https://github.com/ZoeEsther/OCR/assets/119051069/243d1970-2888-428e-8f49-66ef6c7277a9)



