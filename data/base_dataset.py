import torch
import os
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms



class BaseDataset(torch.utils.data.Dataset):
    """docstring for BaseDataset"""
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return os.path.basename(self.opt.data_root.strip('/'))

    def initialize(self, opt):
        self.opt = opt

        self.imgs_dir = os.path.join(self.opt.data_root, self.opt.imgs_dir)
        filename = self.opt.train_csv if self.opt.mode == "train" else self.opt.test_csv
        self.cur_fold = os.path.splitext(filename)[0].split('_')[-1]
        self.imgs_name_file = os.path.join(self.opt.data_root, filename)
        self.imgs_path = self.make_dataset(self.imgs_dir, self.imgs_name_file)

    def make_dataset(self, imgs_dir, imgs_name_file):
        return None

    def load_dict(self, pkl_path):
        saved_dict = {}
        with open(pkl_path, 'rb') as f:
            saved_dict = pickle.load(f, encoding='latin1')
        return saved_dict

    def get_img_by_path(self, img_path):
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        img_type = 'L' if self.opt.img_nc == 1 else 'RGB'
        return Image.open(img_path).convert(img_type)

    def img_transform(self, img, use_data_augment=False, norm_tensor=False, lucky_dict={}):
        if norm_tensor:
            img2tensor = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            img2tensor = transforms.ToTensor()

        img = transforms.functional.resize(img, self.opt.load_size)
        # on-the-fly data augmentation
        if self.opt.mode == "train" and use_data_augment:
            # scale and crop 
            # lucky_num = random.randint(0, 4)
            lucky_num_crop = random.randint(0, 4) if not lucky_dict else lucky_dict['crop']
            img = transforms.functional.five_crop(img, self.opt.final_size)[lucky_num_crop]
            # Horizontally flip
            lucky_num_flip = random.randint(0, 1) if not lucky_dict else lucky_dict['flip']
            if lucky_num_flip:
                img = transforms.functional.hflip(img)
            # update seed dict if needed
            if not lucky_dict:
                lucky_dict.update({'crop': lucky_num_crop, 'flip': lucky_num_flip})
        else:
            img = transforms.functional.five_crop(img, self.opt.final_size)[-1]  # center crop

        # print(lucky_dict)
        return img2tensor(img)

    def __len__(self):
        return len(self.imgs_path)





    







