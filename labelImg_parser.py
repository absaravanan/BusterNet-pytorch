import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import imutils
import xml.etree.ElementTree as ET


data_path = "/home/ai/Downloads/Aadhaar Masking-20190812T070825Z-001/Aadhaar_Masking"
output_data = "/home/ai/ai/data/coco/aadhaar_mask_long/"
onlyfiles = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
onlyfiles = [f for f in onlyfiles if f.endswith('xml')]
fileExtensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']

for index, src_file in enumerate(onlyfiles[:]):

    print (index)
    img_filename = ""
    img_filename, ext = os.path.splitext(src_file)
    only_img_filename = img_filename

    for extn in fileExtensions:
        if os.path.isfile(img_filename+extn):
            img_filename = img_filename+extn
            # print (img_filename)
            continue


    root = ET.parse(src_file).getroot()

    img = cv2.imread(img_filename)
    h, w = img.shape[:2]
    blank_img = np.zeros((h,w,3), np.uint8)
    for child1 in root:
        for child2 in child1:
            # print (child2.tag, child2.attrib)
            if child2.tag == "bndbox":
                for child3 in child2:
                    # print (child3.tag, child3.text)
                    if child3.tag == "xmin":
                        xmin = child3.text
                    if child3.tag == "ymin":
                        ymin = child3.text
                    if child3.tag == "xmax":
                        xmax = child3.text
                    if child3.tag == "ymax":
                        ymax = child3.text

                cv2.rectangle(blank_img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,255,255),-1)
    # cv2.imshow("1", blank_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow("2", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # vis = np.concatenate((blank_img, img), axis=1)
    # vis = cv2.resize(vis, (512,512))

    # cv2.imshow("3", vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    angles = [0, 90, 180, 270]
    for angle in angles:
        rot_img = imutils.rotate_bound(img, angle)
        rot_blank_img = imutils.rotate_bound(blank_img, angle)
        # print (output_data+"images/"+only_img_filename+str(angle)+".jpg")

        cv2.imwrite(output_data+"images/"+only_img_filename.split('/')[-1]+str(angle)+"_l_.jpg", rot_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100] )
        cv2.imwrite(output_data+"annotations/"+only_img_filename.split('/')[-1]+str(angle)+"_l_.jpg", rot_blank_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100] )
