from .base_dataset import BaseDataset
import torch
import os
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms



class CKPlusResDataset(BaseDataset):
    """docstring for CKPlusResDataset"""
    def __init__(self):
        super(CKPlusResDataset, self).__init__()
        
    def initialize(self, opt):
        super(CKPlusResDataset, self).initialize(opt)
        # load facial expression dictionary 
        cls_pkl = os.path.join(self.opt.data_root, self.opt.cls_pkl)
        self.cls_dict = self.load_dict(cls_pkl)
        # init residual map folders 
        self.imgs_res_dir = os.path.join(self.opt.data_root, self.opt.imgs_res_dir, str(self.cur_fold))

    def get_cls_by_path(self, img_path):
        sub_seq = os.path.splitext(os.path.basename(img_path))[0]
        sub_seq = '_'.join(sub_seq.split('_')[:2])
        cls_label = self.cls_dict[sub_seq] - 1  # convert [1, 7] to [0, 6] for CK+ dataset label way
        return cls_label

    def get_img_res_by_cls(self, img_cls):
        img_path = os.path.join(self.imgs_res_dir, "%d.png" % img_cls)
        return self.get_img_by_path(img_path)

    def make_dataset(self, imgs_dir, imgs_name_file):
        # do not use offline data augmentation
        imgs = []
        assert os.path.isfile(imgs_name_file), "File '%s' does not exist." % imgs_name_file
        with open(imgs_name_file, 'r') as f:
            lines = f.readlines()
            imgs = [os.path.join(imgs_dir, line.strip()) for line in lines]
            imgs = sorted(imgs)
        return imgs

    def __getitem__(self, index):
        data_dict = {}

        img_path = self.imgs_path[index]
        data_dict['img_path'] = img_path

        real_cls = self.get_cls_by_path(img_path)
        data_dict['real_cls'] = real_cls

        lucky_dict = {}
        # [0, 1] color 
        img_tensor = self.img_transform(self.get_img_by_path(img_path), self.opt.use_data_augment, norm_tensor=False, lucky_dict=lucky_dict)
        data_dict['img_tensor'] = img_tensor

        # [0, 1] gray
        img_tensor_gray = self.img_transform(self.get_img_by_path(img_path).convert('L'), self.opt.use_data_augment, norm_tensor=False, lucky_dict=lucky_dict)
        data_dict['img_tensor_gray'] = img_tensor_gray

        # [0, 1] gray
        img_res_tensor = self.img_transform(self.get_img_res_by_cls(real_cls).convert('L'), self.opt.use_data_augment, norm_tensor=False, lucky_dict=lucky_dict)
        data_dict['img_res_tensor'] = img_res_tensor

        return data_dict


