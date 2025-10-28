import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import kornia
import torch.cuda.amp as amp


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        pred = torch.clamp(pred, min=1e-6)
        target = torch.clamp(target, min=1e-6)
        valid_mask = valid_mask.detach()
        # diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        diff_log = torch.log(pred[valid_mask]) - torch.log(target[valid_mask])

        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss

class RelativeL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def normalize_disparity(self, depth, mask):
        """
        depth: Tensor (H,W) hoặc (B,H,W)
        mask: Boolean tensor same shape as depth
        """
        disp = torch.zeros_like(depth)
        valid_disp = depth[mask]
        disp[mask] = 1.0 / (valid_disp + self.eps)

        # normalize disparity trên vùng hợp lệ
        disp_valid = disp[mask]
        min_disp = torch.quantile(disp_valid, 0.01)
        max_disp = torch.quantile(disp_valid, 0.99)
        disp[mask] = torch.clamp((disp[mask] - min_disp) / (max_disp - min_disp + self.eps), 0.0, 1.0)
        return disp

    def forward(self, pred, target, mask):
        """
        pred, target: (B,H,W) hoặc (H,W)
        mask: Boolean tensor same shape
        """
        valid_mask = mask & (target > 0)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred_n = self.normalize_disparity(pred, valid_mask)
        target_n = self.normalize_disparity(target, valid_mask)

        loss = torch.mean(torch.abs(pred_n[valid_mask] - target_n[valid_mask]))
        return loss


    
class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss,self).__init__()
        self.mse = nn.MSELoss()
        self.grad_factor = 10.
        self.normal_factor = 1.

       
    def forward(self,criterion,pred,target,epoch=0):
        if 'l1' in criterion:
            depth_loss = self.L1_imp_Loss(pred,target)
        elif 'l2' in criterion:
            depth_loss = self.L2_imp_Loss(pred,target)
        elif 'rmsle' in criterion:
            depth_loss = self.RMSLELoss(pred,target)
        if 'gn' in criterion:
            grad_target, grad_pred = self.imgrad_yx(target), self.imgrad_yx(pred)
            grad_loss = self.GradLoss(grad_pred, grad_target)     * self.grad_factor
            normal_loss = self.NormLoss(grad_pred, grad_target) * self.normal_factor
            return depth_loss + grad_loss + normal_loss
        else:
            return depth_loss
        
        # ===== Relative L1 (disparity normalized) =====
    def normalize_disparity(self, depth, eps=1e-6):
        disp = 1.0 / (depth + eps)
        disp = (disp - disp.min()) / (disp.max() - disp.min() + eps)
        return disp

    def L1_imp_Loss(self, pred, target):
        # chỉ xét vùng hợp lệ
        valid_mask = (target > 0).detach()
        pred = pred[valid_mask]
        target = target[valid_mask]

        # normalize disparity trên từng ảnh
        pred_n = self.normalize_disparity(pred)
        target_n = self.normalize_disparity(target)

        loss = torch.mean(torch.abs(pred_n - target_n))
        return loss
    # =============================================
    
    def GradLoss(self,grad_target,grad_pred):
        return torch.sum( torch.mean( torch.abs(grad_target-grad_pred) ) )
    
    def NormLoss(self, grad_target, grad_pred):
        prod = ( grad_pred[:,:,None,:] @ grad_target[:,:,:,None] ).squeeze(-1).squeeze(-1)
        pred_norm = torch.sqrt( torch.sum( grad_pred**2, dim=-1 ) )
        target_norm = torch.sqrt( torch.sum( grad_target**2, dim=-1 ) ) 
        return 1 - torch.mean( prod/(pred_norm*target_norm) )
    
    def RMSLELoss(self, pred, target):
        return torch.sqrt(self.mse(torch.log(pred + 0.5), torch.log(target + 0.5)))
 
        
    
    def L1_imp_Loss(self, pred, target):
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
    
    def L2_imp_Loss(self, pred, target):
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss
    
    def imgrad_yx(self,img):
        N,C,_,_ = img.size()
        grad_y, grad_x = self.imgrad(img)
        return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)
    
    def imgrad(self,img):
        img = torch.mean(img, 1, True)
        fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv1.weight = nn.Parameter(weight)
        grad_x = conv1(img)

        fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv2.weight = nn.Parameter(weight)
        grad_y = conv2(img)
        return grad_y, grad_x
    

class ScaleInvariantGradientMatchingLoss(nn.Module):
    def __init__(self, scales=(1,2,4), loss_type='l1', eps=1e-6):
        super().__init__()
        self.scales = scales
        self.loss_type = loss_type
        self.eps = eps

        kernel_x = torch.tensor([[[[-1, 1]]]], dtype=torch.float32)
        kernel_y = torch.tensor([[[[-1], [1]]]], dtype=torch.float32)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def _downsample(self, x, factor):
        if factor == 1:
            return x
        h = x.shape[-2] // factor
        w = x.shape[-1] // factor
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

    def _gradient_xy(self, x):
        pad_x = (1, 0, 0, 0)
        pad_y = (0, 0, 1, 0)
        gx = F.conv2d(F.pad(x, pad_x, mode='replicate'),
                      self.kernel_x.repeat(x.shape[1], 1, 1, 1),
                      groups=x.shape[1])
        gy = F.conv2d(F.pad(x, pad_y, mode='replicate'),
                      self.kernel_y.repeat(x.shape[1], 1, 1, 1),
                      groups=x.shape[1])
        return gx, gy

    def forward(self, pred, target, valid_mask):
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if valid_mask is None:
            valid_mask = torch.ones_like(pred, dtype=torch.bool, device=pred.device)
        else:
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)
            valid_mask = valid_mask.to(torch.bool)

        pred = pred.clamp(min=self.eps)
        target = target.clamp(min=self.eps)

        log_pred_full = torch.log(pred)
        log_target_full = torch.log(target)

        total_loss = 0.0
        for factor in self.scales:
            lp = self._downsample(log_pred_full, factor)
            lt = self._downsample(log_target_full, factor)
            vm = self._downsample(valid_mask.float(), factor) >= 0.5

            gx_p, gy_p = self._gradient_xy(lp)
            gx_t, gy_t = self._gradient_xy(lt)

            diff_x = gx_p - gx_t
            diff_y = gy_p - gy_t

            if self.loss_type == 'l1':
                per_px = torch.abs(diff_x) + torch.abs(diff_y)
            else:
                per_px = diff_x ** 2 + diff_y ** 2

            mask = vm.expand_as(per_px)
            valid_count = mask.float().sum()
            if valid_count.item() > 0:
                loss_scale = (per_px * mask.float()).sum() / (valid_count + self.eps)
                total_loss += loss_scale

        return total_loss / len(self.scales)


class CombinedDepthLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, lambd_silog=0.5):
        super().__init__()
        self.silog = SiLogLoss(lambd=lambd_silog)
        self.grad_loss = ScaleInvariantGradientMatchingLoss(scales=(1, 2, 4))
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target, valid_mask):
        loss_silog = self.silog(pred, target, valid_mask)
        loss_grad = self.grad_loss(pred, target, valid_mask)
        total = self.alpha * loss_silog + self.beta * loss_grad
        return total, loss_silog, loss_grad



class CustomLoss(nn.Module):
    def __init__(self, lambd=0.5, height=None, width=None):
        super().__init__()
        self.lambd = lambd
        self.height = height
        self.width = width

    def forward(self, pred, target, valid_mask=None):
        if valid_mask is not None:
            pred = pred[valid_mask]
            target = target[valid_mask]
        
        # tính hiệu: di = target - pred
        di = target - pred
        n = pred.numel() if (self.height is None or self.width is None) else (self.height * self.width)
        
        di2 = torch.pow(di, 2)
        first_term = torch.sum(di2) / n
        second_term = self.lambd * torch.pow(torch.sum(di), 2) / (n ** 2)
        
        loss = first_term - second_term
        return loss



class L1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, mask):
        """
        pred, target: (B,H,W) hoặc (H,W)
        mask: Boolean tensor same shape
        """
        valid_mask = mask.detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
    
class L1NormLoss(nn.Module):
    def __init__(self):
        super(L1NormLoss, self).__init__()

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        valid_mask = (target > 0).detach()

        if mask is not None:
            valid_mask *= mask.detach()

        _min, _max = torch.quantile(target[mask].cpu().detach(), torch.tensor([0.02, 1 - 0.02]),)
        gt_depth_norm = (target - _min) / (_max - _min)
        gt_depth_norm = torch.clip(gt_depth_norm, 0.01, 1.0)

        diff = gt_depth_norm - pred
        diff = diff[valid_mask]
        loss = diff.abs().mean()
        return loss
    
class SSILoss(nn.Module):
    """
    Scale shift invariant MAE loss.
    loss = MAE((d-median(d)/s - (d'-median(d'))/s'), s = mean(d- median(d))
    """
    def __init__(self, loss_weight=1, data_type=['sfm', 'stereo', 'lidar'], **kwargs):
        super(SSILoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
    
    def ssi_mae(self, target, prediction, mask):
        valid_pixes = torch.sum(mask) + self.eps

        gt_median = torch.median(target) if target.numel() else 0
        gt_s = torch.abs(target - gt_median).sum() / valid_pixes
        gt_trans = (target - gt_median) / (gt_s + self.eps)

        pred_median = torch.median(prediction) if prediction.numel() else 0
        pred_s = torch.abs(prediction - pred_median).sum() / valid_pixes
        pred_trans = (prediction - pred_median) / (pred_s + self.eps)
        
        ssi_mae_sum = torch.sum(torch.abs(gt_trans - pred_trans))
        return ssi_mae_sum, valid_pixes

    def forward(self, target, prediction, mask=None, **kwargs):
        """
        Calculate loss.
        """
        B, C, H, W = prediction.shape
        loss = 0
        valid_pix = 0
        for i in range(B):
            mask_i = mask[i, ...]
            gt_depth_i = target[i, ...][mask_i]
            pred_depth_i = prediction[i, ...][mask_i]
            ssi_sum, valid_pix_i = self.ssi_mae(pred_depth_i, gt_depth_i, mask_i) 
            loss += ssi_sum
            valid_pix += valid_pix_i
        loss /= (valid_pix + self.eps)
        return loss * self.loss_weight
    

# from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(10)
torch.cuda.manual_seed(10)

class L1NormLoss(nn.Module):
    def __init__(self):
        super(L1NormLoss, self).__init__()

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        valid_mask = (target > 0).detach()

        if mask is not None:
            valid_mask *= mask.detach()

        _min, _max = torch.quantile(target[mask].cpu().detach(), torch.tensor([0.02, 1 - 0.02]),)
        gt_depth_norm = (target - _min) / (_max - _min)
        gt_depth_norm = torch.clip(gt_depth_norm, 0.01, 1.0)

        diff = gt_depth_norm - pred
        diff = diff[valid_mask]
        loss = diff.abs().mean()
        return loss
    
class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        valid_mask = (target > 0).detach()

        if mask is not None:
            valid_mask *= mask.detach()

        diff = target - pred
        diff = diff[valid_mask]
        loss = diff.abs().mean()
        return loss * self.loss_weight

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = target - pred
        diff = diff[valid_mask]
        loss = (diff**2).mean()
        return loss


class BerhuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = torch.abs(target - pred)
        diff = diff[valid_mask]
        delta = self.threshold * torch.max(diff).data.cpu().numpy()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0*delta ** 2, 0.)
        part2 = part2 / (2. * delta)
        diff = part1 + part2
        loss = diff.mean()
        return loss
    
class Silog_Loss(nn.Module):
    def __init__(self, variance_focus=0.85, loss_weight=1.0):
        super(Silog_Loss, self).__init__()
        self.variance_focus = variance_focus
        self.loss_weight = loss_weight

    def forward(self, target, pred, mask=None):
        d = torch.log(pred[mask]) - torch.log(target[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0 * self.loss_weight
    
class RMSELog(nn.Module):
    def __init__(self):
        super(RMSELog, self).__init__()

    def forward(self, target, pred, mask=None):
        #assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()
        target = target[valid_mask]
        pred = pred[valid_mask]
        log_error = torch.abs(torch.log(target / (pred + 1e-12)))
        loss = torch.sqrt(torch.mean(log_error**2))
        return loss

class SSILoss(nn.Module):
    """
    Scale shift invariant MAE loss.
    loss = MAE((d-median(d)/s - (d'-median(d'))/s'), s = mean(d- median(d))
    """
    def __init__(self, loss_weight=1, data_type=['sfm', 'stereo', 'lidar'], **kwargs):
        super(SSILoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
    
    def ssi_mae(self, target, prediction, mask):
        valid_pixes = torch.sum(mask) + self.eps

        gt_median = torch.median(target) if target.numel() else 0
        gt_s = torch.abs(target - gt_median).sum() / valid_pixes
        gt_trans = (target - gt_median) / (gt_s + self.eps)

        pred_median = torch.median(prediction) if prediction.numel() else 0
        pred_s = torch.abs(prediction - pred_median).sum() / valid_pixes
        pred_trans = (prediction - pred_median) / (pred_s + self.eps)
        
        ssi_mae_sum = torch.sum(torch.abs(gt_trans - pred_trans))
        return ssi_mae_sum, valid_pixes

    def forward(self, target, prediction, mask=None, **kwargs):
        """
        Calculate loss.
        """
        B, C, H, W = prediction.shape
        loss = 0
        valid_pix = 0
        for i in range(B):
            mask_i = mask[i, ...]
            gt_depth_i = target[i, ...][mask_i]
            pred_depth_i = prediction[i, ...][mask_i]
            ssi_sum, valid_pix_i = self.ssi_mae(pred_depth_i, gt_depth_i, mask_i) 
            loss += ssi_sum
            valid_pix += valid_pix_i
        loss /= (valid_pix + self.eps)
        return loss * self.loss_weight
    
def gradient_log_loss(log_prediction_d, log_gt, mask):
    log_d_diff = log_prediction_d - log_gt

    v_gradient = torch.abs(log_d_diff[:, :, :-2, :] - log_d_diff[:, :, 2:, :])
    v_mask = torch.mul(mask[:, :, :-2, :], mask[:, :, 2:, :])
    v_gradient = torch.mul(v_gradient, v_mask)

    h_gradient = torch.abs(log_d_diff[:, :, :, :-2] - log_d_diff[:, :, :, 2:])
    h_mask = torch.mul(mask[:, :, :, :-2], mask[:, :, :, 2:])
    h_gradient = torch.mul(h_gradient, h_mask)

    EPSILON = 1e-6
    N = torch.sum(h_mask) + torch.sum(v_mask) + EPSILON

    gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
    gradient_loss = gradient_loss / N

    return gradient_loss
    
class GradientLoss_Li(nn.Module):
    def __init__(self, scale_num=4, loss_weight=1, data_type = ['lidar', 'stereo'], **kwargs):
        super(GradientLoss_Li, self).__init__()
        self.__scales = scale_num
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6

    def forward(self, target, prediction, mask, **kwargs):
        total = 0
        target_trans = target + (~mask) * 100
        pred_log = torch.log(prediction)
        gt_log = torch.log(target_trans)
        for scale in range(self.__scales):
            step = pow(2, scale)
            
            total += gradient_log_loss(pred_log[:, ::step, ::step], gt_log[:, ::step, ::step], mask[:, ::step, ::step])
        loss = total / self.__scales
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            return 0 * torch.sum(prediction)
            # raise RuntimeError(f'VNL error, {loss}')
        return loss * self.loss_weight
    
class EPNLoss(nn.Module):
    """
    Hieratical depth spatial normalization loss for Gaussian sampling.
    Replace the original grid masks with the random created masks.
    loss = MAE((d-median(d)/s - (d'-median(d'))/s'), s = mean(d- median(d))
    """
    def __init__(self, loss_weight=1.0, random_num=32, batch_limit=8, lower_bound=0.125, upper_bound=0.5, **kwargs):
        super(EPNLoss, self).__init__()
        self.loss_weight = loss_weight
        self.random_num = random_num
        self.batch_limit = batch_limit
        self.eps = 1e-6
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_random_masks_for_batch(self, image_size: list)-> torch.Tensor:
        height, width = image_size
        crop_h_min = int(self.lower_bound * height)
        crop_h_max = int(self.upper_bound * height)
        crop_w_min = int(self.lower_bound * height)
        crop_w_max = int(self.upper_bound * height)
        h_max = height - crop_h_min
        w_max = width - crop_w_min
        crop_height = np.random.choice(np.arange(crop_h_min, crop_h_max), self.random_num, replace=False)
        crop_width = np.random.choice(np.arange(crop_w_min, crop_w_max), self.random_num, replace=False)
        crop_y = np.clip(np.random.normal(h_max / 2, h_max / 6, self.random_num).astype(int), 0, h_max - 1)
        crop_x = np.random.choice(w_max, self.random_num, replace=False)
        crop_y_end = crop_height + crop_y
        crop_y_end[crop_y_end>=height] = height
        crop_x_end = crop_width + crop_x

        mask_new = torch.zeros((self.random_num, height, width), dtype=torch.bool, device="cuda") #.cuda() #[N, H, W]
        for i in range(self.random_num):
            if crop_x_end[i] <= width:
                mask_new[i, crop_y[i]:crop_y_end[i], crop_x[i]:crop_x_end[i]] = True
            else:
                mask_new[i, crop_y[i]:crop_y_end[i], crop_x[i]:width] = True
                mask_new[i, crop_y[i]:crop_y_end[i], 0:(crop_x_end[i] - width)] = True

        return mask_new
  
    def ssi_mae(self, prediction, target, mask_valid):
        B, C, H, W = target.shape
        prediction_nan = prediction.clone().detach()
        target_nan = target.clone()
        prediction_nan[~mask_valid] = float('nan')
        target_nan[~mask_valid] = float('nan')

        valid_pixs = mask_valid.reshape((B, C,-1)).sum(dim=2, keepdims=True) + 1e-10
        valid_pixs = valid_pixs[:, :, :, None]

        gt_median = target_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # [b,c,h,w]
        gt_median[torch.isnan(gt_median)] = 0
        gt_diff = (torch.abs(target - gt_median) ).reshape((B, C, -1))
        gt_s = gt_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        gt_trans = (target - gt_median) / (gt_s + self.eps)

        pred_median = prediction_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # [b,c,h,w]
        pred_median[torch.isnan(pred_median)] = 0
        pred_diff = (torch.abs(prediction - pred_median)).reshape((B, C, -1))
        pred_s = pred_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        pred_trans = (prediction - pred_median) / (pred_s + self.eps)

        loss_sum = torch.sum(torch.abs(gt_trans - pred_trans)*mask_valid)
        return loss_sum

    def forward(self, target, prediction, mask=None, sem_mask=None, **kwargs):
        """
        Calculate loss.
        """
        B, C, H, W = target.shape
        
        loss = 0.0
        valid_pix = 0.0

        device = target.device
        
        self.batch_valid = torch.tensor([1], device=device)[:,None,None,None]

        batch_limit = self.batch_limit
        
        random_sample_masks = self.get_random_masks_for_batch((H, W)) # [N, H, W]
        for i in range(B):
            # each batch
            mask_i = mask[i, ...] #[1, H, W]
            pred_i = prediction[i, ...].unsqueeze(0).repeat(batch_limit, 1, 1, 1)
            target_i = target[i, ...].unsqueeze(0).repeat(batch_limit, 1, 1, 1)
            random_sem_masks = random_sample_masks

            sampled_masks_num = random_sem_masks.shape[0]
            loops = int(np.ceil(sampled_masks_num / batch_limit))

            for j in range(loops):
                mask_random_sem_loopi = random_sem_masks[j*batch_limit:(j+1)*batch_limit, ...]
                mask_sample = (mask_i & mask_random_sem_loopi).unsqueeze(1) # [N, 1, H, W]
                loss += self.ssi_mae(
                    prediction=pred_i[:mask_sample.shape[0], ...], 
                    target=target_i[:mask_sample.shape[0], ...], 
                    mask_valid=mask_sample)
                valid_pix += torch.sum(mask_sample)
        
        # the whole image
        mask = mask * self.batch_valid.bool()
        loss += self.ssi_mae(
                    prediction=prediction, 
                    target=target, 
                    mask_valid=mask)
        valid_pix += torch.sum(mask)
        loss = loss / (valid_pix + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            print(f'HDSNL NAN error, {loss}, valid pix: {valid_pix}')
        return loss * self.loss_weight