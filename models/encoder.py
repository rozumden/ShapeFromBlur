import torch
import torch.nn as nn

def mesh_normalize(vertices):
    mesh_max = torch.max(vertices, dim=1, keepdim=True)[0]
    mesh_min = torch.min(vertices, dim=1, keepdim=True)[0]
    mesh_middle = (mesh_min + mesh_max) / 2
    vertices = vertices - mesh_middle
    bs = vertices.shape[0]
    mesh_biggest = torch.max(vertices.view(bs, -1), dim=1)[0]
    vertices = vertices / mesh_biggest.view(bs, 1, 1) # * 0.45
    return vertices

class EncoderBasic(nn.Module):
    def __init__(self, config, ivertices, faces, face_features, width, height):
        super(EncoderBasic, self).__init__()
        self.config = config
        self.translation = nn.Parameter(torch.zeros(1,config["input_frames"],6))
        self.quaternion = nn.Parameter(torch.ones(1,config["input_frames"],8))
        if self.config["predict_vertices"]:
            self.vertices = nn.Parameter(torch.zeros(1,ivertices.shape[0],3))
        if self.config["texture_size"] > 0:
            self.register_buffer('face_features', torch.from_numpy(face_features).unsqueeze(0).type(self.translation.dtype))
            self.face_features_oper = lambda x: x
            self.texture_map = nn.Parameter(torch.ones(1,3,self.config["texture_size"],self.config["texture_size"]))
        else:
            self.face_features = nn.Parameter(torch.ones(1,faces.shape[0],3,3))
            self.face_features_oper = nn.Sigmoid()
            self.texture_map = None
        ivertices = torch.from_numpy(ivertices).unsqueeze(0).type(self.translation.dtype)
        ivertices = mesh_normalize(ivertices)
        self.register_buffer('ivertices', ivertices)
        self.aspect_ratio = height/width

    def forward(self):
        thr = self.config["camera_distance"]-2
        thrn = thr*4
        translation_all = []
        quaternion_all = []
        for frmi in range(self.translation.shape[1]):
            translation = nn.Tanh()(self.translation[:,frmi,None,:])
            quaternion = self.quaternion[:,frmi]
           
            translation = translation.view(translation.shape[:2]+torch.Size([1,2,3]))
            translation_new = translation.clone()
            translation_new[:,:,:,:,2][translation[:,:,:,:,2] > 0] = translation[:,:,:,:,2][translation[:,:,:,:,2] > 0]*thr
            translation_new[:,:,:,:,2][translation[:,:,:,:,2] < 0] = translation[:,:,:,:,2][translation[:,:,:,:,2] < 0]*thrn
            translation_new[:,:,:,:,:2] = translation[:,:,:,:,:2]*( (self.config["camera_distance"]-translation_new[:,:,:,:,2:])/2 )
            translation = translation_new
            translation[:,:,:,:,1] = self.aspect_ratio*translation_new[:,:,:,:,1]
            translation[:,:,:,0,:] = translation[:,:,:,0,:] - translation[:,:,:,1,:]

            quaternion = quaternion.view(quaternion.shape[:1]+torch.Size([1,2,4]))
            
            translation_all.append(translation)
            quaternion_all.append(quaternion)

        translation = torch.stack(translation_all,2).contiguous()[:,:,:,0]
        quaternion = torch.stack(quaternion_all,1).contiguous()[:,:,0]
        if self.config["predict_vertices"]:
            vertices = self.ivertices + self.vertices
            if self.config["mesh_normalize"]:
                vertices = mesh_normalize(vertices)
            else:
                vertices = vertices - vertices.mean(1)[:,None,:] ## make center of mass in origin
        else:
            vertices = self.ivertices

        face_features_out = self.face_features_oper(self.face_features)

        return translation, quaternion, vertices, face_features_out, self.texture_map

