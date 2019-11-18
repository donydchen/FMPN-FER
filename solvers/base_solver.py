"""
Created on Dec 13, 2018
@author: Yuedong Chen
"""

from data import create_dataloader
from model import create_model
from visualizer import Visualizer
import copy



class BaseSolver(object):
    """docstring for BaseSolver"""
    def __init__(self):
        super(BaseSolver, self).__init__()

    def initialize(self, opt):
        self.opt = opt
        self.visual = Visualizer()
        self.visual.initialize(self.opt)

        self.CK_FACIAL_EXPRESSION = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        self.OC_FACIAL_EXPRESSION = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

    def run_solver(self):
        if self.opt.mode == "train":
            self.train_networks()
        else:
            self.test_networks(self.opt)

    def train_networks(self):
        # init train setting
        self.init_train_setting()

        # for every epoch
        for epoch in range(self.opt.epoch_count, self.epoch_len + 1):
            # train network
            self.train_epoch(epoch)
            # update learning rate
            self.cur_lr = self.train_model.update_learning_rate()
            # save checkpoint if needed
            if epoch % self.opt.save_epoch_freq == 0:
                self.train_model.save_ckpt(epoch)

        # save the last epoch 
        self.train_model.save_ckpt(self.epoch_len)

    def init_train_setting(self):
        self.train_dataset = create_dataloader(self.opt)
        self.train_model = create_model(self.opt)
        if 'CK' in self.train_dataset.name() or 'Affect' in self.train_dataset.name():
            self.FACIAL_EXPRESSION = self.CK_FACIAL_EXPRESSION
        else:
            self.FACIAL_EXPRESSION = self.OC_FACIAL_EXPRESSION

        self.train_total_steps = 0
        self.epoch_len = self.opt.niter + self.opt.niter_decay
        self.cur_lr = self.opt.lr

    def train_epoch(self, epoch):
        pass

    def test_networks(self, opt):
        pass

    def init_test_setting(self, opt):
        # hard code some params
        opt.visdom_display_id = 0
        opt.serial_batches = True

        dataset = create_dataloader(opt)
        model = create_model(opt)
        model.set_eval()
        return dataset, model




