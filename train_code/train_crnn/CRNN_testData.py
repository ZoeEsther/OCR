################ 用测试集测试 CRNN ####
import train_code.train_crnn.mydataset as mydataset
from train_code.train_crnn.crnn_recognizer import PytorchOcr
from train_code.train_crnn.online_test import val_on_image
import config
import cv2

# config.test_infofile = 'D:/OCR/train_code/train_crnn/test_data/test.txt'


def test(infofile):

    model_path = 'D:/OCR/checkpoints/CRNN-6-3.pth'
    recognizer = PytorchOcr(model_path)

    with open(infofile) as f:
        content = f.readlines()
        num_all = 0
        num_correct = 0

        for line in content:
            if '.jpg ' in line:
                fname, label = line.split('.jpg ')
                fname = "E:/CRNNtest/data/" + fname + ".jpg"
            else:
                fname, label = line.split('g:')
                fname += 'g'
            label = label.replace('\r', '').replace('\n', '')
            img = cv2.imread(fname)
            h, w = img.shape[:2]
            res = recognizer.recognize(img)
            res = res.strip()
            label = label.strip()


            print("########### %d ##########" %num_all)
            print("pred : ", res)
            print("label: ", label)
            if res == label:
                num_correct+=1
            num_all+=1

        acc = float( num_correct/num_all)

        print("----- acc ----- : ",acc)

if __name__ == "__main__":

    infofile = 'E:/CRNNtest/test.txt'
    test(infofile)