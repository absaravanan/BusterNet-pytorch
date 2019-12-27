import cv2
import argparse
from model import BusterNetCore

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
import traceback

def plot_img_and_mask(img, mask, src_mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 3, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 3, 2)
    b.set_title('Output mask')
    plt.imshow(mask)

    c = fig.add_subplot(1,3, 3)
    c.set_title('Src mask')
    plt.imshow(src_mask)

    plt.show()

def overlay_mask(img, mask):
    print (img.shape, mask.shape)
    # res = cv2.bitwise_and(img,img,mask = mask)
    img[mask==0] = (0,0,255)
    return img


def predict_on_image(model, args):
    # pre-processing on image
    image_dir = "/home/ryzen/ai/data/copy_move/val/pan"
    out_files = []
    mask_files = []
    for r, d, f in os.walk(image_dir):
        for filethis in f:
            # print (os.path.join(r,file))
            if "forged_image.jpg" in filethis:
                out_files.append(os.path.join(r,filethis))
            if "source_target_mask.jpg" in filethis:
                mask_files.append(os.path.join(r,filethis))
    model.eval()
    
    # out_files.sort()
    # mask_files.sort()
    
    for i, fn in enumerate(out_files[:]):
        # torch.cuda.empty_cache()
        try:
            image = cv2.imread(fn, -1)
            h, w = image.shape[:2]
            if h > 1000 or w > 1000:
                mask = cv2.imread(mask_files[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

                resize = iaa.Scale({'height': args.crop_height, 'width': args.crop_width})
                resize_det = resize.to_deterministic()
                image = resize_det.augment_image(image)

                image = Image.fromarray(image).convert('RGB')

                img_src = image.copy()
                image = transforms.ToTensor()(image)
                image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image).unsqueeze(0)
                # read csv label path
                label_info = get_label_info(args.csv_path)
                # predict
                st = time.time()
                predict = model(image.cpu()).squeeze()
                predict = reverse_one_hot(predict)
                print (hash(predict))
                print ("predicted")
                print (predict)
                unique, counts = np.unique(predict, return_counts=True)
                print (dict(zip(unique, counts)))
                # predict = colour_code_segmentation(np.array(predict.cpu()), label_info)
                # predict = cv2.resize(np.uint8(predict), (960, 720))
                # cv2.imwrite(args.save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
                # img_src[np.uint8(predict)!=0] = (0,0,255)
                # img_src.show()
                predict = np.array(predict.cpu(), dtype=np.int32)
                print (type(predict))
                print (predict.shape)

                img = np.array(img_src, dtype=np.int32)
                print (img.shape)
                

                print (time.time() - st)
                # img = img.astype(int)
                # print (type(img), type(predict))
                # try:
                # w, h = img.shape[:2]
                # predict = np.resize(predict, (500,500))
                # img = np.resize(img, (500,500))
                # mask = np.resize(mask, (500,500))

                # predict = np.resize(predict,(w,h))
                # print (type(img), type(predict))



                # img = overlay_mask(img, predict)
                # img.show()
                # predict = np.expand_dims(predict, -1)
                plot_img_and_mask(img_src, predict, mask)

        except Exception as e:
            print (e)
            traceback.print_exc()
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
    parser.add_argument('--crop_height', type=int, default=256, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=256, help='Width of cropped/resized input image to network')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=False, help='Whether to user gpu for training')
    parser.add_argument('--csv_path', type=str, default=None, required=True, help='Path to label info csv file')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='Path to save predict image')


    args = parser.parse_args(params)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    # model = BiSeNet(args.num_classes, args.context_path)
    model = BusterNetCore.SimiNet()
    from torchsummary import summary
    print (model)
    net = summary(model, (3, 256, 256))
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
        '--checkpoint_path', 'checkpoints_18_sgd/latest_dice_loss3.pth',
        '--csv_path', 'dataset/class_dict.csv',
        '--save_path', 'demo.png',
        '--context_path', 'resnet18'
    ]
    main(params)