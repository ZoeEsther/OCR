import os
from ocr import ocr
import cv2
import shutil
import numpy as np
from PIL import Image
from glob import glob
from detect.ctpn_predict import get_det_boxes
from ocr import sort_box


def single_pic_proc(image_file):
    image = np.array(Image.open(image_file).convert('RGB'))
    result, image_framed = ocr(image)
    return result,image_framed


if __name__ == '__main__':

    image_files = glob('D:/CTPN_TestCode/CTPN_testData/*.*')  #
    result_dir = 'D:/CTPN_TestCode/CTPN_testcode_result'  #
    for image_file in sorted(image_files):
        image = np.array(Image.open(image_file).convert('RGB'))
        text, image, img = get_det_boxes(image)

        text = sort_box(text)

        text = np.trunc(text).astype(int).tolist()
        # text = text.astype(int)
        # text = np.matrix.tolist(text)
        # print(text)

        output_file = os.path.join(result_dir + '/'+ image_file.split('\\')[-1])
        txt_file = os.path.join(result_dir + '/' + 'res_'+ image_file.split('\\')[-1].split('.')[0] + '.txt')

        print(txt_file)
        Image.fromarray(image).save(output_file)

        with open(txt_file, 'w') as txt_f:
            for tag in text:
                t=tag[4]
                tag[4]=tag[6]
                tag[6]=t

                a = tag[5]
                tag[5] = tag[7]
                tag[7] = a

                for i in tag[:8]:
                    txt_f.write(str(i)+",")
                txt_f.write('\n')





    # image_files = glob('D:/OCR/detect/data/*.*') #
    # result_dir = 'D:/OCR/detect/data_result'#
    # for image_file in sorted(image_files):
    #     t = time.time()
    #     result, image_framed = single_pic_proc(image_file)
    #     output_file = os.path.join(result_dir, image_file.split('/')[-1])
    #     txt_file = os.path.join(result_dir, image_file.split('/')[-1].split('.')[0]+'.txt')
    #
    #     print(txt_file)
    #     txt_f = open(txt_file, 'w')
    #     Image.fromarray(image_framed).save(output_file)
    #     print("Mission complete, it took {:.3f}s".format(time.time() - t))
    #     print("\nRecognition Result:\n")
    #     for key in result:
    #         print(result[key][1])
    #         txt_f.write(result[key][1]+'\n')
    #     txt_f.close()

