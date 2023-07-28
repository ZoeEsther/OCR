# OCR
This project realizes the recognition of product identification cahracters based on CRNN and CTPN, mainly involving English alphabet and Punctuation. It includes a visual GUI interface that allows for adaptive adjustment of overexposed or dark images.


-------------------- 中文 --------------------
开发软件工具版本
    训练环境: PyCharm+Python3.7+GPU
    测试环境: PyCharm+Python3.7+CPU
    主要Packages及对应版本:
	                     torch       	                1.8.1
	                     torchvision	                0.9.1
	                     OpenCv-python           	    4.5.1.48
	                     numpy	                      1.16.6
                       cv                           1.0.0
                       matplotlib                   3.3.2
                       pandas                       1.1.4
                       trans                        2.1.0

代码功能
     a. 文件夹train_code：包含train_ctpn、train_crnn两个文件夹，前者为文本检测模型CTPN训练文件夹，后者为文字识别模型CRNN的训练文件夹。
         将模型训练需要的训练数据放入对应的train_data（CRNN网络模型还需要验证集val_data)，运行ctpn_train.py、train_pytorch_ctc.py即开始训练模型。
         其他.py文件均内含训练模型所需的函数。
     b. 文件夹checkpoints：包含网络模型的训练结果；
     c. 文件夹detect：文本检测CTPN测试代码；
     d. 文件夹recognize：文字识别CRNN测试代码；
     e. 文件夹GUI：界面代码；
     f. 文件夹HSV、Median、sharpness等其他均为图像处理代码；
     g. demo.py是将文本检测与文字识别结合在一起的有完整字符识别功能的代码；


-------------------- English --------------------
Code Function
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

Configuration environment
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
        
