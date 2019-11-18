"""
Created on Dec 12, 2018
@author: Yuedong Chen
"""

import torch
import os
from collections import OrderedDict
import random

from .base_model import BaseModel
from . import model_utils



class ResGenModel(BaseModel):
    """docstring for ResGenModel"""
    def __init__(self):
        super(ResGenModel, self).__init__()
         
    def initialize(self, opt):
        super(ResGenModel, self).initialize(opt)
        self.net_resface = model_utils.define_ResFaceGenNet(input_nc=1, img_size=self.opt.final_size, ngf=64, norm=self.opt.res_norm, \
                                use_dropout=self.opt.res_use_dropout, n_blocks=self.opt.res_n_blocks, \
                                init_type=self.opt.init_type, init_gain=self.opt.init_gain, gpu_ids=self.gpu_ids)
        self.models_name.append('resface')
    
    def setup(self):
        super(ResGenModel, self).setup()
        if self.is_train:
            self.losses_name.append('resface')

            self.optim_resface = torch.optim.Adam(self.net_resface.parameters(), 
                        lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optims.append(self.optim_resface)

            self.schedulers.append(model_utils.get_scheduler(self.optim_resface, self.opt))

        if self.opt.load_epoch > 0:
            self.load_ckpt(self.opt.load_epoch)
        
    def feed_batch(self, batch):
        self.real_img = batch['img_tensor'].to(self.device)
        self.real_img_path = batch['img_path']
        if self.is_train:
            self.real_cls = batch['real_cls'].type(torch.LongTensor).to(self.device)
            self.real_resface = batch['img_res_tensor'].to(self.device)

    def forward(self):
        self.gen_resface, self.resface_features = self.net_resface(self.real_img)

    def backward(self):
        self.loss_resface = self.criterionMSE(self.gen_resface, self.real_resface)
        self.loss_total = self.loss_resface
        self.loss_total.backward()

    def optimize_paras(self):
        self.forward()
        self.optim_resface.zero_grad()
        self.backward()
        self.optim_resface.step()

    def save_ckpt(self, epoch):
        save_models_name = ['resface']
        return super(ResGenModel, self).save_ckpt(epoch, save_models_name)

    def load_ckpt(self, epoch):
        load_models_name = ['resface']
        return super(ResGenModel, self).load_ckpt(epoch, load_models_name)

    def clean_ckpt(self, epoch):
        clean_models_name = ['resface']
        return super(ResGenModel, self).clean_ckpt(epoch, clean_models_name)

    def get_latest_losses(self):
        losses_name = ['resface']
        return super(ResGenModel, self).get_latest_losses(losses_name)

    def get_latest_visuals(self):
        visuals_name = ['real_img', 'gen_resface']
        if self.is_train:
            visuals_name.append('real_resface')
        return super(ResGenModel, self).get_latest_visuals(visuals_name)
        