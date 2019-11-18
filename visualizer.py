import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import os
import numpy as np
import torch
import math
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Visualizer(object):
    """docstring for Visualizer"""
    def __init__(self):
        super(Visualizer, self).__init__()

    def initialize(self, opt):
        self.opt = opt
        self.vis_saved_dir = os.path.join(self.opt.ckpt_dir, 'vis_pics')
        if not os.path.isdir(self.vis_saved_dir):
            os.makedirs(self.vis_saved_dir)
        plt.switch_backend('agg')
        self.plt_color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        self.display_id = self.opt.visdom_display_id
        if self.display_id > 0:
            import visdom 
            self.ncols = 4
            self.vis = visdom.Visdom(server="http://localhost", port=self.opt.visdom_port, env=self.opt.visdom_env)

    def throw_visdom_connection_error(self):
        print('\n\nno visdom server.')
        exit(1)

    def print_losses_info(self, info_dict):
        msg = '[{}][Epoch: {:0>3}/{:0>3}; Images: {:0>4}/{:0>4}; Time: {:.3f}s/Batch({}); LR: {:.7f}] '.format(
                self.opt.name, info_dict['epoch'], info_dict['epoch_len'], 
                info_dict['epoch_steps'], info_dict['epoch_steps_len'], 
                info_dict['step_time'], self.opt.batch_size, info_dict['cur_lr'])
        for k, v in info_dict['losses'].items():
            msg += '| {}: {:.4f} '.format(k, v)
        msg += '|'
        print(msg)
        with open(info_dict['log_path'], 'a+') as f:
            f.write(msg + '\n')

    def display_current_losses(self, epoch, counter_ratio, losses_dict):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses_dict.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses_dict[k] for k in self.plot_data['legend']])
        try:
            accum_x = np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1)
            accum_y = np.array(self.plot_data['Y'])
            # fix shape checking bug on visdom version '0.1.8.9'
            if accum_y.ndim == 2 and accum_y.shape[1] == 1:
                accum_y = accum_y.reshape(accum_y.shape[0])
                accum_x = accum_x.reshape(accum_x.shape[0])

            self.vis.line(
                X=accum_x,
                Y=accum_y,
                opts={
                    'title': self.opt.name + ' loss over time',
                    'legend':self.plot_data['legend'],
                    'xlabel':'epoch',
                    'ylabel':'loss'},
                win=self.display_id)
        except ConnectionError:
            self.throw_visdom_connection_error()

    def display_cls_acc(self, epoch, acc_dict):
        win_id = self.display_id + 6
        if not hasattr(self, 'plot_acc_data'):
            self.plot_acc_data = {'X': [], 'Y': [], 'legend': list(acc_dict.keys())}
        self.plot_acc_data['X'].append(epoch)
        self.plot_acc_data['Y'].append([acc_dict[k] for k in self.plot_acc_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_acc_data['X'])] * len(self.plot_acc_data['legend']), 1),
                Y=np.array(self.plot_acc_data['Y']),
                opts={
                    'title': 'predict class accuracy over time',
                    'legend':self.plot_acc_data['legend'],
                    'xlabel':'epoch',
                    'ylabel':'acc /%'},
                win=win_id)
        except ConnectionError:
            self.throw_visdom_connection_error()

    def display_cls_confusion_matrix(self, confusion_mat, labels, epoch, name):
        win_id = self.display_id + 4 if name == 'test' else self.display_id + 5
        color_map = 'Oranges' if name == 'test' else 'Blues'
        title = "[%s][%03d]: Confusion Matrix" % (name, epoch)
        df_cm = pd.DataFrame(confusion_mat, index = labels, columns = labels)
        plt.figure(figsize = (5,4))
        sn.heatmap(df_cm, annot=True, cmap=color_map, fmt='g')
        try:
            self.vis.matplot(plt, win=win_id, opts=dict(title=title))
            save_name = os.path.join(self.vis_saved_dir, 'con_mat_%s_%s.png' % (name, str(epoch)))
            plt.savefig(save_name, bbox_inches='tight')
            plt.close()
        except ConnectionError:
            self.throw_visdom_connection_error()

    def display_features_distribution(self, features_dict, label_legend, epoch, name):
        win_id = self.display_id + 14 if name == 'test' else self.display_id + 15
        title = "[%s][%03d]: Features Distribution" % (name, epoch)

        features = np.array(features_dict['features'])
        labels = np.array(features_dict['labels'])
        pca = PCA(n_components=40)
        pca_features = pca.fit_transform(features)
        tsne_embedded = TSNE(n_components=2).fit_transform(pca_features)

        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        for label_idx in range(len(label_legend)):
            plt.scatter(
                features[labels==label_idx, 0],
                features[labels==label_idx, 1],
                c=colors[label_idx],
                s=2,
            )
        plt.legend(label_legend, loc='upper right')
        try:
            self.vis.matplot(plt, win=win_id, opts=dict(title=title))
            save_name = os.path.join(self.vis_saved_dir, 'fet_dis_%s_%s.png' % (name, str(epoch)))
            plt.savefig(save_name, bbox_inches='tight')
            plt.close()
        except ConnectionError:
            self.throw_visdom_connection_error()

    def display_online_results(self, visuals, epoch):
        win_id = self.display_id + 24
        images = []
        labels = []
        for label, image in visuals.items():
            # if 'res' in label:  # or 'focus' in label:
            #     image = (image - 0.5) / 0.5   # convert map from [0, 1] to [-1, 1]
            image_numpy = self.tensor2im(image)
            images.append(image_numpy.transpose([2, 0, 1]))
            labels.append(label)
        try:
            title = '-'.join(labels)
            self.vis.images(images, nrow=self.ncols, win=win_id,
                            padding=5, opts=dict(title=title))
        except ConnectionError:
            self.throw_visdom_connection_error()
        
    def display_offline_results(self, results_dict, epoch, name='train'):
        win_id = self.display_id + 34 if name == 'train' else self.display_id + 35
        labels = list(results_dict.keys())
        imgs_len = len(results_dict[labels[0]])
        images = []
        for i in range(imgs_len):
            for label in labels:
                cur_img = results_dict[label][i]
                # if 'res' in label:  # or 'focus' in label:
                #     cur_img = (cur_img - 0.5) / 0.5   # convert map from [0, 1] to [-1, 1]
                cur_img = self.numpy2im(cur_img).transpose([2, 0, 1])
                images.append(cur_img)
        try:
            title = "[%5s][%03d] %s" % (name, epoch, ' | '.join(labels))
            self.vis.images(images, nrow=4*len(labels), win=win_id,
                            padding=5, opts=dict(title=title))
        except ConnectionError:
            self.throw_visdom_connection_error()

    # utils
    def tensor2im(self, input_image, imtype=np.uint8):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        return self.numpy2im(image_numpy, imtype)
        
    def numpy2im(self, image_numpy, imtype=np.uint8):
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))  
        # input should be [0, 1]
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        # print(image_numpy.shape)
        image_numpy = image_numpy.astype(imtype)
        im = Image.fromarray(image_numpy).resize((64, 64), Image.ANTIALIAS)
        return np.array(im)





