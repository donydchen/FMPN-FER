from .base_dataset import BaseDataset
import torch
import os
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms



class AffectNetDataset(BaseDataset):
    """docstring for AffectNetDataset"""
    def __init__(self):
        super(AffectNetDataset, self).__init__()

    def initialize(self, opt):
        super(AffectNetDataset, self).initialize(opt)
        # load facial expression dictionary 
        cls_pkl = os.path.join(self.opt.data_root, self.opt.cls_pkl)
        self.cls_dict = self.load_dict(cls_pkl)

    def make_dataset(self, imgs_dir, imgs_name_file):
        # specify dataset parsing here
        imgs = []
        assert os.path.isfile(imgs_name_file), "File '%s' does not exist." % imgs_name_file
        with open(imgs_name_file, 'r') as f:
            lines = f.readlines()
            imgs = [os.path.join(imgs_dir, line.strip()) for line in lines]
            imgs = sorted(imgs)
        return imgs

    def get_cls_by_path(self, img_path):
        img_name = os.path.basename(img_path)
        cls_label = self.cls_dict[img_name] -1 

        return cls_label

    def __getitem__(self, index):
        data_dict = {}

        img_path = self.imgs_path[index]
        data_dict['img_path'] = img_path

        img_tensor = self.img_transform(self.get_img_by_path(img_path), self.opt.use_data_augment)
        data_dict['img_tensor'] = img_tensor

        real_cls = self.get_cls_by_path(img_path)
        data_dict['real_cls'] = real_cls

        return data_dict
