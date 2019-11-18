import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from collections import OrderedDict
import torchvision.models as torch_models


'''
Helper functions for model
Borrow tons of code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
'''


# tempoary fix none norm bug 
class PseudoNorm(nn.Module):
    """docstring for PseudoNorm"""
    def __init__(self, num_features):
        super(PseudoNorm, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x 
        

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = functools.partial(PseudoNorm)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_optimizer(params, opt):
    if opt.optim_policy == 'adam':
        optimizer = torch.optim.Adam(params, 
                        lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.optim_policy == 'sgd':
        optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.9)
    elif opt.optim_policy == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=opt.lr)
    else:
        return NotImplementedError('optimizer policy [%s] is not implemented', opt.optim_policy)
    
    return optimizer

def init_weights(net, init_type='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], backend_pretrain=False):
    if len(gpu_ids) > 0:
        # print("gpu_ids,", gpu_ids)
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if not backend_pretrain:
        init_weights(net, init_type, gain=init_gain)
    return net


def custom_inception_v3(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        n_classes = 7
        if 'num_classes' in kwargs:
            n_classes = kwargs['num_classes']
            kwargs['num_classes'] = 1000
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = torch_models.inception.Inception3(**kwargs)
        pretrained_state_dict = torch.utils.model_zoo.load_url('https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth')
        # load only existing feature
        pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model.state_dict()}
        model.load_state_dict(pretrained_dict)
        print("[Info] Successfully load ImageNet pretrained parameters for inception v3.")
        # update fc layer
        print("Predict Class Number:", n_classes, "; Transfrom Input:", kwargs['transform_input'])
        model.fc = nn.Linear(2048, n_classes)
        init.xavier_normal_(model.fc.weight.data, 0.02)
        init.constant_(model.fc.bias.data, 0.0)

        return model

    return torch_models.inception.Inception3(**kwargs)


def define_ClassifierNet(input_nc, image_size=128, n_classes=7, norm='batch', use_dropout=False, init_type='xavier', init_gain=0.02, gpu_ids=[], backend='default', backend_pretrain=False):
    norm_layer = get_norm_layer(norm_type=norm)
    print("[Info] Norm type for Cls net is %s." % norm)
    if backend == 'inception':
        # input size 299
        net_cls = custom_inception_v3(pretrained=backend_pretrain, num_classes=n_classes, aux_logits=False, transform_input=True)
    elif backend == 'resnet50':
        # input size 224
        net_cls = torch_models.resnet50(pretrained=backend_pretrain, num_classes=n_classes)
    elif backend == 'resnet152':
        net_cls = torch_models.resnet152(pretrained=backend_pretrain, num_classes=n_classes)
    elif backend == 'densenet121':
        net_cls = torch_models.densenet121(pretrained=backend_pretrain, num_classes=n_classes)
    else:
        raise NotImplementedError('Classifier backend [%s] is not implemented' % backend)
    return init_net(net_cls, init_type, init_gain, gpu_ids, backend_pretrain)


def define_ResFaceGenNet(input_nc, img_size, ngf=64, norm='batch', use_dropout=False, n_blocks=4, init_type='xavier', init_gain=0.02, gpu_ids=[]):
    norm_type = get_norm_layer(norm_type=norm)
    print("[Info] Norm type for Residual Face Genearating Net is %s." % norm)
    net_resface = ResFaceGenNet(input_nc, img_size, ngf, norm_layer=norm_type, use_dropout=use_dropout, n_blocks=n_blocks, padding_type='zero')
    return init_net(net_resface, init_type, init_gain, gpu_ids)


def define_FusionNet(input_a_nc, input_b_nc, init_type, init_gain, gpu_ids):
    net_fusion = FusionNet(input_a_nc, input_b_nc)
    return init_net(net_fusion, init_type, init_gain, gpu_ids)


#-----------------------------------------------------------------------------------------
# Define Classes 
#-----------------------------------------------------------------------------------------
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.PReLU()]  # change from ReLU to PReLU
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.PReLU()]  # add activation

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# ------------------------------- Residual Face Genearating Net --------------------------------------
class ResFaceGenNet(nn.Module):
    """ Created on Dec 13, 2018
        @author: Yuedong Chen 
        based on Johnson architecture"""
    def __init__(self, input_nc, img_size=299, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='zero'):
        assert (n_blocks >= 0)
        super(ResFaceGenNet, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf
        trans_conv_output_padding = img_size % 2
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Conv2d(self.input_nc, ngf, kernel_size=7, stride=1, padding=3, 
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, \
                                kernel_size=4, stride=2, padding=1, \
                                bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, \
                                    use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2, padding=1,
                                         bias=use_bias, output_padding=trans_conv_output_padding),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        self.feature_extractor = nn.Sequential(*model)

        # learn a residual face as attention map on top of feature extractor
        res_top = [nn.Conv2d(ngf, 1, kernel_size=7, stride=1, padding=3, bias=False),
                    nn.Sigmoid()]
        self.res_top = nn.Sequential(*res_top)

    def forward(self, img):
        embed_features = self.feature_extractor(img)
        res_map = self.res_top(embed_features)
        if self.input_nc > 1:
            res_map = res_map.expand(res_map.size(0), self.input_nc, res_map.size(2), res_map.size(3))
        return res_map, embed_features
# ------------------------------- End Residual Face Genearating Net --------------------------------------


class FusionNet(nn.Module):
    """ Created on Dec 17, 2018
        @author: Yuedong Chen """
    def __init__(self, input_a_nc=3, input_b_nc=1, n_blocks=1, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(FusionNet, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.process_a = nn.Sequential(
                            nn.Conv2d(input_a_nc, 3, kernel_size=3, stride=1, padding=1, bias=use_bias),
                            norm_layer(3),
                            nn.ReLU(True))
        self.process_b = nn.Sequential(
                            nn.Conv2d(input_b_nc, 3, kernel_size=3, stride=1, padding=1, bias=use_bias),
                            norm_layer(3),
                            nn.ReLU(True))
        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(3, padding_type='zero', norm_layer=norm_layer, \
                                    use_dropout=use_dropout, use_bias=use_bias)]
        # fit the input scale of inception v3
        model += [nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.Sigmoid()]  
        self.model = nn.Sequential(*model)

    def forward(self, img_a, img_b):
        pre_a = self.process_a(img_a)
        pre_b = self.process_b(img_b)
        return self.model(pre_a + pre_b)




        