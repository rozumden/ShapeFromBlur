import torch
import torch.nn as nn
import kaolin
from models.kaolin_wrapper import *
from kornia.geometry.conversions import angle_axis_to_rotation_matrix, quaternion_to_angle_axis, rotation_matrix_to_quaternion
from kornia.geometry.conversions import quaternion_to_rotation_matrix, rotation_matrix_to_angle_axis
from kornia.morphology import erosion, dilation
from kornia.filters import GaussianBlur2d
from utils import *

class RenderingKaolin(nn.Module):
    def __init__(self, config, faces, width, height):
        super().__init__()
        self.config = config
        self.height = height
        self.width = width
        camera_proj = kaolin.render.camera.generate_perspective_projection(1.57/2, self.width/self.height ) # 45 degrees
        self.register_buffer('camera_proj', camera_proj)
        self.register_buffer('camera_trans', torch.Tensor([0,0,self.config["camera_distance"]])[None])
        self.register_buffer('obj_center', torch.zeros((1,3)))
        camera_up_direction = torch.Tensor((0,1,0))[None]
        camera_rot,_ = kaolin.render.camera.generate_rotate_translate_matrices(self.camera_trans, self.obj_center, camera_up_direction)
        self.register_buffer('camera_rot', camera_rot)
        self.set_faces(faces)
            
    def set_faces(self, faces):
        self.register_buffer('faces', torch.LongTensor(faces))

    def forward(self, translation, quaternion, unit_vertices, face_features, texture_maps=None):
        kernel = torch.ones(self.config["erode_renderer_mask"], self.config["erode_renderer_mask"]).to(translation.device)
        
        all_renders = []
        for frmi in range(quaternion.shape[1]):
            if quaternion.shape[-1] == 4:
                rotation_matrix = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(quaternion[:,frmi,1]))    
                rotation_matrix_step = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(quaternion[:,frmi,0])/self.config["fmo_steps"]/2)
            else:
                rotation_matrix = quaternion[:,frmi,1]
                rotation_matrix_step = quaternion[:,frmi,0]

            renders = []
            for ti in torch.linspace(0,1,self.config["fmo_steps"]):
                vertices = kaolin.render.camera.rotate_translate_points(unit_vertices, rotation_matrix, self.obj_center) 
                vertices = vertices + translation[:,:,frmi,1] + ti*translation[:,:,frmi,0]

                face_vertices_cam, face_vertices_image, face_normals = prepare_vertices(vertices, self.faces, self.camera_rot, self.camera_trans, self.camera_proj)
                face_vertices_z = face_vertices_cam[:,:,:,-1]
                face_normals_z = face_normals[:,:,-1]
                ren_features, ren_mask, red_index = kaolin.render.mesh.dibr_rasterization(self.height, self.width, face_vertices_z, face_vertices_image, face_features, face_normals_z, sigmainv=self.config["sigmainv"], boxlen=0.02, knum=30, multiplier=1000)
                if not texture_maps is None:
                    ren_features = kaolin.render.mesh.texture_mapping(ren_features, texture_maps, mode='bilinear')
                result = ren_features.permute(0,3,1,2)
                if self.config["erode_renderer_mask"] > 0:
                    ren_mask = erosion(ren_mask[:,None], kernel)[:,0]
                if self.config["apply_blur_inside"] > 0:
                    gauss = GaussianBlur2d((11, 11), (self.config["apply_blur_inside"], self.config["apply_blur_inside"]))
                    result_rgba = torch.cat((gauss(result),gauss(gauss(ren_mask[:,None].float()))),1)
                else:
                    result_rgba = torch.cat((result,ren_mask[:,None]),1)
                renders.append(result_rgba)

                rotation_matrix = torch.matmul(rotation_matrix, rotation_matrix_step)

            renders = torch.stack(renders,1).contiguous()
            all_renders.append(renders)
        all_renders = torch.stack(all_renders,1).contiguous()
        return all_renders
