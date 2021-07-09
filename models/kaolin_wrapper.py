import kaolin
import meshio
import meshzoo
import plyfile
import numpy as np
from models.initial_mesh import generate_initial_mesh
import torch

import pyvista as pv

def load_obj(path):
    return kaolin.io.obj.import_mesh(path,with_materials=True)

def get_lapsmooth(vertices, faces):
	nver = kaolin.metrics.trianglemesh.uniform_laplacian_smoothing(vertices, faces)
	return nver

def get_ael(vertices, faces):
	ael = kaolin.metrics.trianglemesh.average_edge_length(vertices, faces)
	return ael

def prepare_vertices(vertices, faces, camera_rot, camera_trans, camera_proj):
    vertices_camera = kaolin.render.camera.rotate_translate_points(vertices, camera_rot, camera_trans)
    vertices_image = kaolin.render.camera.perspective_camera(vertices_camera, camera_proj)
    face_vertices_camera = kaolin.ops.mesh.index_vertices_by_faces(vertices_camera, faces)
    face_vertices_image = kaolin.ops.mesh.index_vertices_by_faces(vertices_image, faces)
    face_normals = kaolin.ops.mesh.face_normals(face_vertices_camera, unit=True)
    return face_vertices_camera, face_vertices_image, face_normals

def subdivide_mesh(vertices, faces, config, divide_scale = 1, subfilter = 'loop'):
	cells = 3*np.ones((faces.shape[0],4),'int64')
	cells[:,1:] = faces
	mesh = pv.PolyData(vertices, cells)
	mesh2 = mesh.subdivide(divide_scale, subfilter)
	subdivided_vertices = np.array(mesh2.points)
	subdivided_faces = np.array(mesh2.faces).reshape( int(mesh2.faces.shape[0]/4), 4)[:,1:]

	ivertices, ifaces, iface_features = generate_initial_mesh(config["mesh_size"])
	icells = 3*np.ones((ifaces.shape[0],4),'int64')
	icells[:,1:] = ifaces
	imesh = pv.PolyData(ivertices, icells)
	imesh2 = imesh.subdivide(divide_scale, subfilter)
	subdivided_face_features = generate_face_features(np.array(imesh2.points), subdivided_faces)

	return subdivided_vertices, subdivided_faces, subdivided_face_features

def write_obj_mesh(vertices, faces, face_features, name):
	file = open(name,"w")
	file.write("mtllib model.mtl\n")
	file.write("o FMO\n")
	for ver in vertices:
		file.write("v {:.6f} {:.6f} {:.6f} \n".format(ver[0],ver[1],ver[2]))
	for ffeat in face_features:
		for feat in ffeat:
			if len(feat) == 3:
				file.write("vt {:.6f} {:.6f} {:.6f} \n".format(feat[0],feat[1],feat[2]))
			else:
				file.write("vt {:.6f} {:.6f} \n".format(feat[0],feat[1]))
	file.write("usemtl Material.002\n")
	file.write("s 1\n")
	for fi in range(faces.shape[0]):
		fc = faces[fi]+1
		ti = 3*fi + 1
		file.write("f {}/{} {}/{} {}/{}\n".format(fc[0],ti,fc[1],ti+1,fc[2],ti+2))
	file.close() 

def write_mesh(ivertices, ifaces, colors=None, name='tmp.ply'):
	if type(ifaces) is np.ndarray:
		faces = ifaces
	else:
		faces = ifaces.cpu().detach().numpy()
	if type(ivertices) is np.ndarray:
		vertices = ivertices
	else:
		vertices = ivertices[0].cpu().detach().numpy()
	if colors is None:
		cells = [("triangle", faces)]
		mesh = meshio.Mesh(vertices, cells)
		meshio.write(name, mesh)
	else:
		ply_verts = np.empty(len(vertices), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
		ply_verts["x"] = vertices[:, 0]
		ply_verts["y"] = vertices[:, 1]
		ply_verts["z"] = vertices[:, 2]
		ply_verts = plyfile.PlyElement.describe(ply_verts, "vertex")
		if colors is None:
			ply_faces = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
		else:
			ply_faces = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,)), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
			ply_faces["red"] = colors[:,0]*255
			ply_faces["green"] = colors[:,1]*255
			ply_faces["blue"] = colors[:,2]*255
		ply_faces["vertex_indices"] = faces
		ply_faces = plyfile.PlyElement.describe(ply_faces, "face")
		plyfile.PlyData([ply_verts, ply_faces]).write(name)

