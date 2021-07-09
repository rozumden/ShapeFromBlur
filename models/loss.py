import torch.nn as nn
import torch
from kornia import total_variation
import numpy as np

class FMOLoss(nn.Module):
    def __init__(self, config, ivertices, faces):
        super(FMOLoss, self).__init__()
        self.config = config
        if self.config["loss_laplacian"] > 0:    
            self.lapl_loss = LaplacianLoss(ivertices, faces)

    def forward(self, renders, hs_frames, input_batch, translation, quaternion, vertices, texture_maps, faces):
        im_recon_loss = 0
        if self.config["loss_im_recon"]:
            modelled_renders = torch.cat( (renders[:,:,:,:3]*renders[:,:,:,3:], renders[:,:,:,3:4]), 3).mean(2)

            region_of_interest = None
            if self.config["loss_sil_consistency"]:
                region_of_interest = hs_frames[:,:,:,3:].mean(2) > 0.05
            
            im_recon_loss = self.config["loss_im_recon"]*fmo_model_loss(input_batch, modelled_renders, Mask = region_of_interest)
        
        sil_consistency_loss = 0*im_recon_loss
        if self.config["loss_sil_consistency"]+self.config["loss_rgb_weight"]+self.config["loss_jointm_iou_weight"] > 0:
            if hs_frames[:,:,:,3].min() == 1.0:
                hs_frames_renderer = torch.cat((input_batch[:,:,None,3:] * (1 - renders[:,:,:,3:]) + renders[:,:,:,:3], renders[:,:,:,3:]), 3)
                hs_frames_gt = torch.cat((hs_frames[:,:,:,:3], hs_frames[:,:,:,3:]), 3)
            else:
                hs_frames_renderer = renders
                hs_frames_gt = hs_frames
            for frmi in range(hs_frames_gt.shape[1]):
                loss1 = fmo_loss(hs_frames_renderer[:,frmi], hs_frames_gt[:,frmi], self.config)
                sil_consistency_loss = sil_consistency_loss + loss1 / hs_frames_gt.shape[1]

        loss_lap = 0*im_recon_loss
        if self.config["predict_vertices"] and self.config["loss_laplacian"] > 0:
            loss_lap = self.config["loss_laplacian"]*self.lapl_loss(vertices)

        loss_tv = 0*im_recon_loss
        if self.config["loss_texture_smoothness"] > 0:
            loss_tv = self.config["loss_texture_smoothness"]*total_variation(texture_maps)/(3*self.config["texture_size"]**2)

        loss = im_recon_loss + sil_consistency_loss + loss_lap + loss_tv
        return im_recon_loss, sil_consistency_loss, loss_lap, loss_tv, loss

def fmo_loss(Yp, Y, config):
    YM = Y[:,:,-1:,:,:]
    YpM = Yp[:,:,-1:,:,:]
    YF = Y[:,:,:3]
    YpF = Yp[:,:,:3]
    YMb = ((YM+YpM) > 0).type(YpM.dtype)

    loss = torch.zeros((YM.shape[0],1)).to(Y.device)
   
    if config["loss_sil_consistency"] > 0:
        loss = loss + config["loss_sil_consistency"]*iou_loss(YM, YpM)

    if config["loss_rgb_weight"] > 0:
        loss = loss + config["loss_rgb_weight"]*batch_loss(YpF, YF*YM, YMb[:,:,[0,0,0]], do_mult=False)
    
    if config["loss_jointm_iou_weight"] > 0:
        loss = loss + config["loss_jointm_iou_weight"]*iou_loss(YM.max(1)[0][:,None], YpM.max(1)[0][:,None])

    return loss

def iou_loss(YM, YpM):
    A_inter_B = YM * YpM
    A_union_B = (YM + YpM - A_inter_B)
    iou = 1 - (torch.sum(A_inter_B, [2,3,4]) / torch.sum(A_union_B, [2,3,4])).mean(1)
    return iou

def batch_loss(YpM, YM, YMb, do_mult=True):
    if do_mult:
        losses = nn.L1Loss(reduction='none')(YpM*YMb, YM*YMb)
    else:
        losses = nn.L1Loss(reduction='none')(YpM, YM)
    if len(losses.shape) > 4:
        bloss = losses.sum([1,2,3,4]) / YMb.sum([1,2,3,4])
    else:
        bloss = losses.sum([1,2,3]) / (YMb.sum([1,2,3]) + 0.01)
    return bloss

    
def fmo_model_loss(input_batch, renders, Mask = None):    
    expected = input_batch[:,:,3:] * (1 - renders[:,:,3:]) + renders[:,:,:3]
    if Mask is None:
        Mask = renders[:,:,3:] > 0.05
    Mask = Mask.type(renders.dtype)
    model_loss = 0
    for frmi in range(input_batch.shape[1]):
        temp_loss = batch_loss(expected[:,frmi], input_batch[:,frmi,:3], Mask[:,frmi])
        model_loss = model_loss + temp_loss / input_batch.shape[1]
    return model_loss


# Taken from
# https://github.com/ShichenLiu/SoftRas/blob/master/soft_renderer/losses.py

class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.shape[0]
        self.nf = faces.shape[0]
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            laplacian[i, :] /= laplacian[i, i]

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.matmul(self.laplacian, x)
        dims = tuple(range(x.ndimension())[1:])
        x = x.pow(2).mean(dims)
        if self.average:
            return x.sum() / batch_size
        else:
            return x
        
