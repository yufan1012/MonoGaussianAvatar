import torch
from torch import nn
from model.vgg_feature import VGGPerceptualLoss
from torch.autograd import Variable
from math import exp
from torch.nn import functional as F

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, 3)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window=window, window_size=self.window_size,
                     channel=self.channel, size_average=self.size_average)

class Loss(nn.Module):
    # def __init__(self, mask_weight, var_expression=None, lbs_weight=0,sdf_consistency_weight=0, eikonal_weight=0, vgg_feature_weight=0):
    def __init__(self, var_expression=None, lbs_weight=0, sdf_consistency_weight=0, eikonal_weight=0,
                 vgg_feature_weight=0):
        super().__init__()
        
        # self.mask_weight = mask_weight
        self.lbs_weight = lbs_weight
        self.sdf_consistency_weight = sdf_consistency_weight
        self.eikonal_weight = eikonal_weight
        self.vgg_feature_weight = vgg_feature_weight
        self.var_expression = var_expression
        if self.var_expression is not None:
            self.var_expression = self.var_expression.unsqueeze(1).expand(1, 3, -1).reshape(1, -1).cuda()
        print("Expression variance: ", self.var_expression)

        if self.vgg_feature_weight > 0:
            self.get_vgg_loss = VGGPerceptualLoss().cuda()

        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='none')
        self.ssim_loss = SSIM()

    def get_rgb_loss(self, rgb_values, rgb_gt, weight=None):
        if weight is not None:
            image = rgb_values
            image_gt = rgb_gt
            ssim_loss = self.ssim_loss(image.permute(0, 3, 1, 2),
                                       image_gt.reshape(rgb_gt.shape[0], image.shape[1], image.shape[2], 3).permute(0, 3, 1, 2))
            Lssim = 1.0 - ssim_loss
            rgb_loss = self.l1_loss(rgb_values.reshape(-1, 3) * weight.reshape(-1, 1), rgb_gt.reshape(-1, 3) * weight.reshape(-1, 1))
        else:
            image = rgb_values
            image_gt = rgb_gt
            ssim_loss = self.ssim_loss(image.permute(0, 3, 1, 2),
                                       image_gt.reshape(rgb_gt.shape[0], image.shape[1], image.shape[2], 3).permute(0, 3, 1, 2))
            Lssim = 1.0 - ssim_loss
            rgb_loss = self.l1_loss(rgb_values.reshape(-1, 3), rgb_gt.reshape(-1, 3))
        return rgb_loss, Lssim

    def get_lbs_loss(self, lbs_weight, gt_lbs_weight, use_var_expression=False):
        # the same function is used for lbs, shapedirs, posedirs.
        if use_var_expression and self.var_expression is not None:
            lbs_loss = torch.mean(self.l2_loss(lbs_weight, gt_lbs_weight) / self.var_expression / 50)
        else:
            lbs_loss = self.l2_loss(lbs_weight, gt_lbs_weight).mean()
        return lbs_loss

    def get_mask_loss(self, predicted_mask, object_mask):
        mask_loss = self.l1_loss(predicted_mask.reshape(-1).float(), object_mask.reshape(-1).float())
        return mask_loss

    def get_gt_blendshape(self, index_batch, flame_lbs_weights, flame_posedirs, flame_shapedirs, ghostbone):
        if ghostbone:
            gt_lbs_weight = torch.zeros(len(index_batch), 6).cuda()
            gt_lbs_weight[:, 1:] = flame_lbs_weights[index_batch, :]
        else:
            gt_lbs_weight = flame_lbs_weights[index_batch, :]

        gt_shapedirs = flame_shapedirs[index_batch, :, 100:]
        # print(22222)
        # print(gt_shapedirs.size())
        # print(flame_shapedirs.size())
        gt_posedirs = torch.transpose(flame_posedirs.reshape(36, -1, 3), 0, 1)[index_batch, :, :]

        output = {
            'gt_lbs_weights': gt_lbs_weight,
            'gt_posedirs': gt_posedirs,
            'gt_shapedirs': gt_shapedirs,
        }
        return output

    def get_sdf_consistency_loss(self, sdf_values):
        return torch.mean(sdf_values * sdf_values)

    def get_eikonal_loss(self, grad_theta):
        assert grad_theta.shape[1] == 3
        assert len(grad_theta.shape) == 2
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def forward(self, model_outputs, ground_truth):
        rgb_loss, ssim_loss = self.get_rgb_loss(model_outputs['rgb_image'], ground_truth['rgb'])
        loss = 1 * rgb_loss + 0.25 * ssim_loss

        out = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'ssim_loss': ssim_loss
        }
        if self.vgg_feature_weight > 0:
            bz = model_outputs['batch_size']
            img_res = model_outputs['img_res']
            gt = ground_truth['rgb'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

            predicted = model_outputs['rgb_image'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

            vgg_loss = self.get_vgg_loss(predicted, gt)
            out['vgg_loss'] = vgg_loss
            out['loss'] += vgg_loss * self.vgg_feature_weight

        if self.lbs_weight != 0:
            num_points = model_outputs['lbs_weights'].shape[0]
            ghostbone = model_outputs['lbs_weights'].shape[-1] == 6
            outputs = self.get_gt_blendshape(model_outputs['index_batch'], model_outputs['flame_lbs_weights'],
                                             model_outputs['flame_posedirs'], model_outputs['flame_shapedirs'],
                                             ghostbone)

            lbs_loss = self.get_lbs_loss(model_outputs['lbs_weights'].reshape(num_points, -1),
                                             outputs['gt_lbs_weights'].reshape(num_points, -1),
                                             )

            out['loss'] += lbs_loss * self.lbs_weight * 0.1
            out['lbs_loss'] = lbs_loss

            gt_posedirs = outputs['gt_posedirs'].reshape(num_points, -1)
            posedirs_loss = self.get_lbs_loss(model_outputs['posedirs'].reshape(num_points, -1) * 10,
                                              gt_posedirs* 10,
                                              )
            out['loss'] += posedirs_loss * self.lbs_weight * 10.0
            out['posedirs_loss'] = posedirs_loss
            gt_shapedirs = outputs['gt_shapedirs'].reshape(num_points, -1)
            shapedirs_loss = self.get_lbs_loss(model_outputs['shapedirs'].reshape(num_points, -1)[:, :50*3] * 10,
                                               gt_shapedirs * 10,
                                               use_var_expression=True,
                                               )
            out['loss'] += shapedirs_loss * self.lbs_weight * 10.0
            out['shapedirs_loss'] = shapedirs_loss

        return out