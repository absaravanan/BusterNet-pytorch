import torch
import glob
import os
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from utils import get_label_info, one_hot_it, RandomCrop, reverse_one_hot, one_hot_it_v11, one_hot_it_v11_dice
import random
import matplotlib.pyplot as plt
from PIL import ImageFilter

def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass


def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass

class CamVid(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, csv_path, scale, loss='dice', mode='train'):
        super().__init__()
        self.mode = mode
        self.image_list = []
        if not isinstance(image_path, list):
            image_path = [image_path]
        for image_path_ in image_path:
            self.image_list.extend(glob.glob(os.path.join(image_path_, '*.jpg')))
        self.image_list.sort()
        self.label_list = []
        if not isinstance(label_path, list):
            label_path = [label_path]
        for label_path_ in label_path:
            self.label_list.extend(glob.glob(os.path.join(label_path_, '*.jpg')))
        self.label_list.sort()
        print (len( self.label_list))
        # self.image_name = [x.split('/')[-1].split('.')[0] for x in self.image_list]
        # self.label_list = [os.path.join(label_path, x + '_L.png') for x in self.image_list]
        self.fliplr = iaa.Fliplr(0.5)
        self.label_info = get_label_info(csv_path)
        # resize
        # self.resize_label = transforms.Resize(scale, Image.NEAREST)
        # self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        # normalization
        self.to_tensor = transforms.Compose([
            transforms.ColorJitter(brightness = 2, contrast=2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        # self.crop = transforms.RandomCrop(scale, pad_if_needed=True)
        self.image_size = scale
        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        self.loss = loss

    def __getitem__(self, index):
        # load image and crop
        seed = random.random()
        img = Image.open(self.image_list[index])
        img = img.convert('RGB')
        if bool(random.getrandbits(1)):
            rand_rad = random.randint(0,3)
            img = img.filter(ImageFilter.GaussianBlur(radius=rand_rad))

        # img.show()
        # print (type(img))
        # open_cv_image = np.array(img) 
        # print (type(img))

        # # Convert RGB to BGR 
        # open_cv_image = open_cv_image[:, :, ::-1].copy()
        # cv2.imshow("1", open_cv_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # random crop image
        # =====================================
        # w,h = img.size
        # th, tw = self.scale
        # i = random.randint(0, h - th)
        # j = random.randint(0, w - tw)
        # img = F.crop(img, i, j, th, tw)
        # =====================================

        scale = random.choice(self.scale)
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

        # randomly resize image and random crop
        # =====================================
        if self.mode == 'train':
            img = transforms.Resize(scale, Image.BILINEAR)(img)
            img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)
        # =====================================

        img = np.array(img)
        # load label
        label = Image.open(self.label_list[index])
        label = label.convert('RGB')
        # label.show()
        # print (type(label))
        # print (label.mode)

        # label = np.stack((label,)*3, axis=-1)
        # print (type(label))
        # print (label.size())


        # label = Image.fromarray(label)
        # print (type(label))
        # print (label.size())


        # crop the corresponding label
        # =====================================
        # label = F.crop(label, i, j, th, tw)
        # =====================================

        # randomly resize label and random crop
        # =====================================
        if self.mode == 'train':
            label = transforms.Resize(scale, Image.NEAREST)(label)
            label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)
        # =====================================

        label = np.array(label)


        # augment image and label
        if self.mode == 'train':
            seq_det = self.fliplr.to_deterministic()
            img = seq_det.augment_image(img)
            label = seq_det.augment_image(label)


        # image -> [C, H, W]
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()


        if self.loss == 'dice':
            # label -> [num_classes, H, W]
            label = one_hot_it_v11_dice(label, self.label_info).astype(np.uint8)

            label = np.transpose(label, [2, 0, 1]).astype(np.float32)
            # label = label.astype(np.float32)
            label = torch.from_numpy(label)

            return img, label

        elif self.loss == 'crossentropy':
            label = one_hot_it_v11(label, self.label_info).astype(np.uint8)
            # label = label.astype(np.float32)
            label = torch.from_numpy(label).long()
            # print (label.shape)
            # label = np.expand_dims(label, axis=0)
            # label = torch.unsqueeze(label, 0)


            return img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    # data = CamVid('/path/to/CamVid/train', '/path/to/CamVid/train_labels', '/path/to/CamVid/class_dict.csv', (640, 640))
    data = CamVid(['/home/ai/ai/data/coco/aadhaar_mask_augmented/train/images', '/home/ai/ai/data/coco/aadhaar_mask/val/images'],
                  ['/home/ai/ai/data/coco/aadhaar_mask_augmented/train/annotations', '/home/ai/ai/data/coco/aadhaar_mask/val/annotations'], '/home/ai/ai/abs/BiSeNet/dataset/class_dict.csv',
                  (720, 960), loss='crossentropy', mode='val')

    # data = CamVid(['/home/ai/Downloads/CamVid/train', '/home/ai/Downloads/CamVid/val'],
    #               ['/home/ai/Downloads/CamVid/train_labels', '/home/ai/Downloads/CamVid/val_labels'], '/home/ai/Downloads/CamVid/class_dict.csv',
    #               (720, 960), loss='crossentropy', mode='val')
    # from model.build_BiSeNet import BiSeNet
    # print (len(data))
    from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy
    print  ("loaded ")

    label_info = get_label_info('/home/ai/ai/abs/BiSeNet/dataset/class_dict.csv')
    for i, (img, label) in enumerate(data):
        print ("1")
        print(label.size())
        print(img.size())
        print ("2")

        print(torch.max(label))

