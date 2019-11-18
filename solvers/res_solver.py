"""
Created on Dec 13, 2018
@author: Yuedong Chen
"""

from .base_solver import BaseSolver 
import time
import os
import torch


class ResFaceSolver(BaseSolver):
    """docstring for ResFaceSolver"""
    def __init__(self):
        super(ResFaceSolver, self).__init__()

    def train_epoch(self, epoch):
        self.train_model.set_train()
        last_print_losses_freq_t = time.time()
        for idx, batch in enumerate(self.train_dataset):
            self.train_total_steps += 1

            self.train_model.feed_batch(batch)
            self.train_model.optimize_paras()

            if self.train_total_steps % self.opt.print_losses_freq == 0:
                cur_losses = self.train_model.get_latest_losses()
                avg_step_t = (time.time() - last_print_losses_freq_t) / self.opt.print_losses_freq
                last_print_losses_freq_t = time.time()

                info_dict = {'epoch': epoch, 'epoch_len': self.epoch_len,
                            'epoch_steps': idx * self.opt.batch_size, 'epoch_steps_len': len(self.train_dataset),
                            'step_time': avg_step_t, 'cur_lr': self.cur_lr,
                            'log_path': os.path.join(self.opt.ckpt_dir, self.opt.log_file),
                            'losses': cur_losses
                            }
                self.visual.print_losses_info(info_dict)
                if self.visual.display_id > 0:
                    self.visual.display_current_losses(epoch - 1, info_dict['epoch_steps'] / len(self.train_dataset), cur_losses)

            if self.train_total_steps % self.opt.sample_img_freq == 0 and self.visual.display_id > 0:
                cur_vis = self.train_model.get_latest_visuals()
                self.visual.display_online_results(cur_vis, epoch)

    def test_networks(self, opt):
        # go through all the dataset and generate map
        dataset, model = self.init_test_setting(opt)
        results_dict = {'real_img': [], 'gen_resface': []}
        for idx, batch in enumerate(dataset):
            with torch.no_grad():
                model.feed_batch(batch)
                model.forward()

                results_dict['real_img'].append(model.real_img[0].cpu().float().numpy())
                results_dict['gen_resface'].append(model.gen_resface[0].cpu().float().numpy())

        return results_dict
        


