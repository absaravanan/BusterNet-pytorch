import cv2
import argparse
from model.build_BiSeNet import BiSeNet
import os
import torch
import cv2
from imgaug import augmenters as iaa
from PIL import Image
from torchvision import transforms
import numpy as np
from utils import reverse_one_hot, get_label_info, colour_code_segmentation
import matplotlib.pyplot as plt
import time
from random import randint
import requests
import fitz
import glob
import shutil
import sys
import imutils
from loguru import logger
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> {level} {message}", filter="my_module", level="INFO", backtrace=True)


# build model
os.environ['CUDA_VISIBLE_DEVICES'] = ""
model = BiSeNet(2, "resnet18")
model = torch.nn.DataParallel(model)

# load pretrained model if exists
model.module.load_state_dict(torch.load("checkpoints_18_sgd/latest_dice_loss22.pth", map_location=torch.device('cpu')))
model.eval()
print('Done!')


def plot_img_and_mask(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(mask)
    plt.show()

def overlay_mask(img, mask):
    # print (img.shape, mask.shape)
    # res = cv2.bitwise_and(img,img,mask = mask)
    img[mask==0] = (0,0,255)
    return img


def predict_on_image(imageFile):
    image = cv2.imread(imageFile, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resize = iaa.Scale({'height': 480, 'width': 640})
    resize_det = resize.to_deterministic()
    image = resize_det.augment_image(image)
    image = Image.fromarray(image).convert('RGB')
    img_src = image.copy()
    image = transforms.ToTensor()(image)
    image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image).unsqueeze(0)
    # read csv label path
    label_info = get_label_info("dataset/class_dict.csv")
    # predict
    st = time.time()
    predict = model(image.cpu()).squeeze()
    predict = reverse_one_hot(predict)
    # print (predict)
    # predict = colour_code_segmentation(np.array(predict.cpu()), label_info)
    # predict = cv2.resize(np.uint8(predict), (960, 720))
    # cv2.imwrite(args.save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
    # img_src[np.uint8(predict)!=0] = (0,0,255)
    # img_src.show()
    predict = np.array(predict.cpu(), dtype=np.int32)
    img = np.array(img_src, dtype=np.int32)
    print (time.time() - st)
    # img = img.astype(int)
    # try:
    # w, h = img.shape[:2]
    # predict = np.resize(predict,(w,h))
    # print (type(img), type(predict))
    # try:
    img = overlay_mask(img, predict)
    # img.show()
    #     plot_img_and_mask(img, predict)
    # except Exception as e:
    #     pass
    return img


def download_file(url):
    local_filename = url.split('/')[-1].split('?')[0]
    local_filename = local_filename.lower()
    org_filename ,file_extension = os.path.splitext(local_filename)
    new_filename = org_filename+file_extension

    r = requests.get(url, stream=True)

    directory = os.path.join(os.getcwd(),"tmp")
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename=os.path.join(os.getcwd(),"tmp/"+new_filename)
    with open(filename, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    return filename

def get_imagefile_from_pdf(pdf_filename):
    doc = fitz.open(pdf_filename)
    folder_name = pdf_filename.replace(".pdf", "")
    os.mkdir(folder_name)
    for i in range(1):
        for img in doc.getPageImageList(i):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 5:       # this is GRAY or RGB
                pix.writePNG(os.path.join(folder_name ,"img{}-{}.png".format(i, xref)))
            else:               # CMYK: convert to RGB first
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                pix1.writePNG(os.path.join(folder_name ,"img{}-{}.png".format(i, xref)))
                pix1 = None
            pix = None

    files = glob.glob(folder_name+ "/*.png")
    files.sort()
    logger.info("| {0} | {1}".format(requestId,files))
    if os.path.isfile(os.path.join(folder_name, files[0])):
        imageNumpy = cv2.imread(os.path.join(folder_name, files[0]))
        imageNumpy = imutils.resize(imageNumpy, width=800)
        return imageNumpy, folder_name
    return str(), str()


if __name__ == '__main__':
    import pandas as pd
    import csv
    import time
    image_dir = "/media/saravanan/aace6202-85e5-4d83-8359-469fe554fc87/universe/apps/utils/tmp2/"
    output_csv_file = "output.csv"
    data_path = "/media/saravanan/aace6202-85e5-4d83-8359-469fe554fc87/universe/Downloads/aadhaar masking - QA - Tiff_ocr_check - 0.5.3 - all_samples.csv"

    # download files
    out_files = []
    for r, d, f in os.walk(image_dir):
        for file in f:
            # print (os.path.join(r,file))
            out_files.append(os.path.join(r,file))
    print (len(out_files))

    # read csv
    df=pd.read_csv(data_path)
    k=(len(df['input']))
    start = time.time()  
    for i in range(0, k):
        print(i)
        # try:
        w, h = img.shape[:2]
        predict = np.resize(predict,(w,h))
        print (type(img), type(predict))
        try:

            img = overlay_mask(img, predict)
            # img.show()
            plot_img_and_mask(img, predict)
        except Exception as e:
            pass


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', action='store_true', default=False, help='predict on image')
    parser.add_argument('--video', action='store_true', default=False, help='predict on video')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='The path to the pretrained weights of model')
    parser.add_argument('--context_path', type=str, default="resnet18", help='The context path model you are using.')
    parser.add_argument('--num_classes', type=int, default=2, help='num of object classes (with void)')
    parser.add_argument('--data', type=str, default=None, help='Path to image or video for prediction')
    parser.add_argument('--crop_height', type=int, default=960, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1280, help='Width of cropped/resized input image to network')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=False, help='Whether to user gpu for training')
    parser.add_argument('--csv_path', type=str, default=None, required=True, help='Path to label info csv file')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='Path to save predict image')


    args = parser.parse_args(params)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    model = BiSeNet(args.num_classes, args.context_path)
    # if torch.cuda.is_available() and args.use_gpu:
    model = torch.nn.DataParallel(model)

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device('cpu')))
    print('Done!')

    # predict on image
    if args.image:
        predict_on_image(model, args)

    # predict on video
    if args.video:
        pass

if __name__ == '__main__':
    params = [
        '--image',
        '--data', '/home/ai/ai/data/coco/aadhaar_mask_augmented_4/train/images/05jawelmNPYVSGProU9lcdCjF6mgBSeTvQZ273wmIM4hZlZ9Vj_270.jpg',
        '--checkpoint_path', './checkpoints_18_sgd/latest_dice_loss6462.pth',
        '--csv_path', '/home/ai/ai/abs/BiSeNet/dataset/class_dict.csv',
        '--save_path', 'demo.png',
        '--context_path', 'resnet18'
    ]
    main(params)
