"""
Created on Dec 14, 2018
@author: Yuedong Chen
"""

import torch
import os
from collections import OrderedDict
import random

from .base_model import BaseModel
from . import model_utils
import torchvision


class ResClsModel(BaseModel):
    """docstring for ResClsModel"""
    def __init__(self):
        super(ResClsModel, self).__init__()

    def initialize(self, opt):
        super(ResClsModel, self).initialize(opt)
        self.net_resface = model_utils.define_ResFaceGenNet(input_nc=1, \
                                img_size=self.opt.final_size, ngf=64, norm=self.opt.res_norm, \
                                use_dropout=self.opt.res_use_dropout, n_blocks=self.opt.res_n_blocks, \
                                init_type=self.opt.init_type, init_gain=self.opt.init_gain, gpu_ids=self.gpu_ids)
        self.models_name.append('resface')

        self.net_fusion = model_utils.define_FusionNet(input_a_nc=3, input_b_nc=1, init_type=self.opt.init_type, \
                                init_gain=self.opt.init_gain, gpu_ids=self.gpu_ids)
        self.models_name.append('fusion')

        self.net_cls = model_utils.define_ClassifierNet(3, image_size=self.opt.final_size, \
                    n_classes=self.opt.cls_nc, norm=self.opt.cls_norm, use_dropout=self.opt.use_cls_dropout, \
                    init_type=self.opt.init_type, init_gain=self.opt.init_gain, gpu_ids=self.gpu_ids, \
                    backend=self.opt.cls_backend, backend_pretrain=self.opt.backend_pretrain)
        self.models_name.append('cls')

    def setup(self):
        super(ResClsModel, self).setup()
        if self.is_train:
            self.losses_name.append('resface')
            self.losses_name.append('cls')

            self.optim_cls = torch.optim.Adam([
                                {'params': self.net_cls.parameters()},
                                {'params': self.net_fusion.parameters()},
                                {'params': self.net_resface.parameters(), 'lr': self.opt.res_lr}
                            ], lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optims.append(self.optim_cls)

            self.schedulers.append(model_utils.get_scheduler(self.optim_cls, self.opt))

        if self.opt.load_epoch > 0:
            self.load_ckpt(self.opt.load_epoch)

    def feed_batch(self, batch):
        self.real_img = batch['img_tensor'].to(self.device)
        self.real_img_gray = batch['img_tensor_gray'].to(self.device)
        self.real_img_path = batch['img_path']
        if self.is_train:
            self.real_cls = batch['real_cls'].type(torch.LongTensor).to(self.device)
            self.real_resface = batch['img_res_tensor'].to(self.device)

    def forward(self):
        self.gen_resface, self.resface_features = self.net_resface(self.real_img_gray)
        self.focus_face = self.gen_resface * self.real_img_gray
        # self.focus_face = ((self.real_img_gray * 0.5 + 0.5) * self.gen_resface - 0.5) / 0.5

        self.cls_input = self.net_fusion(self.real_img, self.focus_face)
        # self.cls_input = self.net_fusion(self.real_img, self.gen_resface)
        
        self.pred_cls = self.net_cls(self.cls_input)
        # self.focus_face = self.focus_face.expand(self.focus_face.size(0), self.opt.img_nc, self.focus_face.size(2), self.focus_face.size(3))
        # self.pred_cls = self.net_cls(self.focus_face)

    def backward(self):
        self.loss_cls = self.criterionCE(self.pred_cls, self.real_cls)
        self.loss_resface = self.criterionMSE(self.gen_resface, self.real_resface)

        self.loss_total = self.loss_cls * self.opt.lambda_cls + \
                            self.loss_resface * self.opt.lambda_resface
        self.loss_total.backward()

    def optimize_paras(self):
        self.forward()
        self.optim_cls.zero_grad()
        self.backward()
        self.optim_cls.step()

    def save_ckpt(self, epoch):
        # save the specific networks
        save_models_name = ['cls', 'resface', 'fusion']
        return super(ResClsModel, self).save_ckpt(epoch, save_models_name)

    def load_ckpt(self, epoch):
        # load the specific part of networks
        load_models_name = ['resface']
        if not self.is_train:
            load_models_name.extend(['fusion', 'cls'])
        return super(ResClsModel, self).load_ckpt(epoch, load_models_name)

    def clean_ckpt(self, epoch):
        # load the specific part of networks
        load_models_name = ['cls', 'resface', 'fusion']
        return super(ResClsModel, self).clean_ckpt(epoch, load_models_name)

    def get_latest_losses(self):
        get_losses_name = ['cls', 'resface']
        return super(ResClsModel, self).get_latest_losses(get_losses_name)

    def get_latest_visuals(self):
        visuals_name = ['real_img', 'gen_resface', 'focus_face']
        if self.is_train:
            visuals_name.append('real_resface')
        return super(ResClsModel, self).get_latest_visuals(visuals_name)



