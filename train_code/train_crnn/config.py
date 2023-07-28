import train_code.train_crnn.keys as keys

train_infofile = 'E://ocr//train_code//train_crnn//train_data//train.txt'
train_infofile_fullimg = ''
val_infofile = 'E://ocr//train_code//train_crnn//train_data//test.txt'
alphabet = keys.alphabet
alphabet_v2 = keys.alphabet_v2
workers = 4
batchSize = 50
imgH = 32
imgW = 280
nc = 1
nclass = len(alphabet)+1
nh = 256
niter = 100
lr = 0.0003
beta1 = 0.5
cuda = True
ngpu = 1
pretrained_model = 'D:/OCR/train_code/train_crnn/crnn_models/CRNN-1010-111.pth'# D:/OCR/checkpoints/CRNN-1010-111.pth # D:/OCR/train_code/train_crnn/crnn_models/CRNN-1-1.pth
saved_model_dir = 'crnn_models'
saved_model_prefix = 'CRNN_5/6'
test_model_path = "D:/OCR/checkpoints/CRNN-1010-111.pth"

use_log = True
remove_blank = False

experiment = None
displayInterval = 500
n_test_disp = 10
valInterval = 500
saveInterval = 500
adam = False
adadelta = False
keep_ratio = False
random_sample = True

