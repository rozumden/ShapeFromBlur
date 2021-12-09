import os
import torch
import numpy as np

from utils import *
from models.initial_mesh import generate_initial_mesh
from models.kaolin_wrapper import load_obj, write_obj_mesh
from torchvision.utils import save_image

from models.encoder import *
from models.rendering import *
from models.loss import *

from kornia.feature import DeFMO
from torchvision import transforms

from scipy.ndimage.filters import gaussian_filter

class ShapeFromBlur():
    def __init__(self, config, device = None):
        self.config = config
        self.device = device
        self.defmo = DeFMO(pretrained=True).to(device)
        self.defmo.train(False)

    def apply(self,I,B,bbox_tight,nsplits,radius,obj_dim):
        g_resolution_x = int(640/2)
        g_resolution_y = int(480/2)
        self.defmo.rendering.tsr_steps = nsplits*self.config["factor"]
        self.defmo.rendering.times = torch.linspace(0.01,0.99,nsplits*self.config["factor"])
        bbox = extend_bbox(bbox_tight.copy(),4.0*np.max(radius),g_resolution_y/g_resolution_x,I.shape)
        im_crop = crop_resize(I, bbox, (g_resolution_x, g_resolution_y))
        bgr_crop = crop_resize(B, bbox, (g_resolution_x, g_resolution_y))
        input_batch = torch.cat((transforms.ToTensor()(im_crop), transforms.ToTensor()(bgr_crop)), 0).unsqueeze(0).float()
        with torch.no_grad():
            renders = self.defmo(input_batch.to(self.device))
        renders_rgba = renders[0].data.cpu().detach().numpy().transpose(2,3,1,0)
        est_hs = rev_crop_resize(renders_rgba,bbox,np.zeros((I.shape[0],I.shape[1],4)))
        
        self.bbox = extend_bbox(bbox_tight.copy(),1.0*np.max(radius),g_resolution_y/g_resolution_x,I.shape)
        im_crop = crop_resize(I, self.bbox, (g_resolution_x, g_resolution_y))
        self.bgr_crop = crop_resize(B, self.bbox, (g_resolution_x, g_resolution_y))
        input_batch = torch.cat((transforms.ToTensor()(im_crop), transforms.ToTensor()(self.bgr_crop)), 0).unsqueeze(0).float()
        defmo_masks = crop_resize(est_hs, self.bbox, (g_resolution_x, g_resolution_y))
        hs_frames = torch.zeros((1,nsplits*self.config["factor"],4,input_batch.shape[-2],input_batch.shape[-1]))
        for tti in range(nsplits*self.config["factor"]):
            hs_frames[0,tti] = transforms.ToTensor()(defmo_masks[:,:,:,tti])
        best_model = self.apply_sfb(input_batch, hs_frames)
        if "hs_frames" in best_model:
            best_model["hs_frames"] = best_model["hs_frames"].reshape(1,1,nsplits,self.config["factor"],4,renders.shape[-2],-1).mean(3)
        best_model["renders"] = best_model["renders"].reshape(1,1,nsplits,self.config["factor"],4,renders.shape[-2],-1).mean(3)
        return best_model

    def apply_sfb(self, input_batch, hs_frames):
        input_batch, hs_frames = input_batch[None].to(self.device), hs_frames[None].to(self.device)
        config = self.config.copy()

        config["fmo_steps"] = hs_frames.shape[2]
        if config["write_results"]:
            save_image(input_batch[0,:,:3],os.path.join(self.config["write_results_folder"],'im.png'))
            save_image(hs_frames[0].view(config["input_frames"]*config["fmo_steps"],4,hs_frames.shape[-2],-1),os.path.join(self.config["write_results_folder"],'renders_hs.png'))

        width = hs_frames.shape[-1]
        height = hs_frames.shape[-2]
        best_model = {}
        best_model["value"] = 100
        for prot in config["shapes"]: 
            if prot == 'sphere':
                ivertices, faces, iface_features = generate_initial_mesh(config["mesh_size"])
            else:
                mesh = load_obj(os.path.join('.','prototypes',prot+'.obj'))
                ivertices = mesh.vertices.numpy()
                faces = mesh.faces.numpy().copy()
                iface_features = mesh.uvs[mesh.face_uvs_idx].numpy()

            torch.backends.cudnn.benchmark = True
            rendering = RenderingKaolin(config, faces, width, height).to(self.device)
            loss_function = FMOLoss(config, ivertices, faces).to(self.device)

            for predict_vertices in config["predict_vertices_list"]:
                config["erode_renderer_mask"] = self.config["erode_renderer_mask"]

                config["predict_vertices"] = predict_vertices
                encoder = EncoderBasic(config, ivertices, faces, iface_features, width, height).to(self.device)

               	if config["verbose"]:
                    print('Total params {}'.format(sum(p.numel() for p in encoder.parameters())))

                all_parameters = list(encoder.parameters())
                optimizer = torch.optim.Adam(all_parameters, lr = config["learning_rate"])

                encoder.train()
                for epoch in range(config["iterations"]):
                    translation, quaternion, vertices, face_features, texture_maps = encoder()
                    renders = rendering(translation, quaternion, vertices, face_features, texture_maps)
                    im_recon_loss, sil_consistency_loss, loss_lap, loss_tv, jloss = loss_function(renders, hs_frames, input_batch, translation, quaternion, vertices, texture_maps, rendering.faces)

                    jloss = jloss.mean()
                    optimizer.zero_grad()
                    jloss.backward()
                    optimizer.step()
                    av_im_recon_loss = im_recon_loss.mean().item()
                    if config["verbose"] and epoch % 20 == 0:
                        print("Epoch {:4d}".format(epoch+1), end =" ")
                        if config["loss_im_recon"]:
                            print(", im recon {:.3f}".format(av_im_recon_loss), end =" ")
                        if config["loss_sil_consistency"]:
                            print(", silh {:.3f}".format(sil_consistency_loss.mean().item()), end =" ")
                        if config["loss_laplacian"] > 0:
                            print(", lap {:.3f}".format(loss_lap.mean().item()), end =" ")
                        if config["loss_texture_smoothness"] > 0:
                            print(", tex {:.3f}".format((loss_tv.mean().item())), end =" ")
                        print(", joint {:.3f}".format(jloss.item()))
                    
                    if epoch == 99:
                        config["erode_renderer_mask"] = 5
                    elif epoch == 199:
                        config["erode_renderer_mask"] = 7
                    elif epoch == 299:
                        config["erode_renderer_mask"] = 11

                    if av_im_recon_loss < best_model["value"]:
                        best_model["value"] = av_im_recon_loss
                        best_model["renders"] = renders.detach().cpu().numpy()

                        if config["write_results"]:
                            best_model["vertices"] = vertices.detach().clone()
                            best_model["texture_maps"] = texture_maps.detach().clone()
                            best_model["translation"] = translation.detach().clone()
                            best_model["quaternion"] = quaternion.detach().clone()
                            best_model["face_features"] = face_features.detach().clone()
                            best_model["faces"] = faces
                            best_model["prototype"] = prot
                            best_model["predict_vertices"] = predict_vertices
                            write_renders(renders, input_batch, hs_frames, config, self.config["write_results_folder"])
                            save_image(best_model["texture_maps"], os.path.join(self.config["write_results_folder"],'tex.png'))
                            
        if config["write_results"]:
            write_renders(renders, input_batch, hs_frames, config, self.config["write_results_folder"])
            write_obj_mesh(best_model["vertices"][0].cpu().numpy(), best_model["faces"], best_model["face_features"][0].cpu().numpy(), os.path.join(self.config["write_results_folder"],'mesh.obj'))
            save_image(best_model["texture_maps"], os.path.join(self.config["write_results_folder"],'tex.png'))
            print("Best model type {}, predict vertices {}".format(best_model["prototype"],best_model["predict_vertices"]))
            best_model["hs_frames"] = hs_frames.detach().cpu().numpy()

        if config["apply_blur_inside"] > 0:
            for ki in range(best_model["renders"].shape[2]): 
                best_model["renders"][0,0,ki,3] = gaussian_filter(best_model["renders"][0,0,ki,3], sigma=3*config["apply_blur_inside"])

        return best_model




def write_renders(renders, input_batch, hs_frames, config, tmp_folder):
    modelled_renders = torch.cat( (renders[:,:,:,:3]*renders[:,:,:,3:4], renders[:,:,:,3:4]), 3).mean(2)
    expected = input_batch[:,:,3:] * (1 - modelled_renders[:,:,3:]) + modelled_renders[:,:,:3]
    expected_hs_frames = input_batch[:,:,None,3:] * (1 - renders[:,:,:,3:4]) + renders[:,:,:,:3]*renders[:,:,:,3:4]
    renders_flipped = torch.flip(renders,[2])
    if ((renders - hs_frames)**2).sum() < ((renders_flipped - hs_frames)**2).sum():
        save_image(renders[0].view(config["input_frames"]*config["fmo_steps"],4,renders.shape[-2],-1),os.path.join(tmp_folder,'renders_rgba.png'))
        save_image(renders[0,:,:,:3].view(config["input_frames"]*config["fmo_steps"],3,renders.shape[-2],-1),os.path.join(tmp_folder,'renders_rgb.png'))
        save_image(renders[0,:,:,3:].view(config["input_frames"]*config["fmo_steps"],1,renders.shape[-2],-1),os.path.join(tmp_folder,'renders_mask.png'))
        save_image(expected_hs_frames[0,0],os.path.join(tmp_folder,'renders_tsr.png'))
    else:
        save_image(renders_flipped[0].view(config["input_frames"]*config["fmo_steps"],4,renders.shape[-2],-1),os.path.join(tmp_folder,'renders_rgba.png'))
        save_image(renders_flipped[0,:,:,:3].view(config["input_frames"]*config["fmo_steps"],3,renders.shape[-2],-1),os.path.join(tmp_folder,'renders_rgb.png'))
        save_image(renders_flipped[0,:,:,3:].view(config["input_frames"]*config["fmo_steps"],1,renders.shape[-2],-1),os.path.join(tmp_folder,'renders_mask.png'))
        save_image(torch.flip(expected_hs_frames,[2])[0,0],os.path.join(tmp_folder,'renders_tsr.png'))
    save_image(expected[0],os.path.join(tmp_folder,'im_recon.png'))