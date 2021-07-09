import meshzoo
import numpy as np

def generate_initial_mesh(meshsize):
	vertices, faces = meshzoo.icosa_sphere(meshsize)
	face_features = generate_face_features(vertices, faces)
	return vertices, faces, face_features

def generate_face_features(vertices, faces):
	face_features = np.zeros([faces.shape[0],3,2])
	for ki in range(faces.shape[0]):
		for pi in range(3):
			ind = faces[ki,pi]
			face_features[ki,pi] = [np.arctan2(-vertices[ind, 0], vertices[ind, 1])/(2 * np.pi), np.arcsin(vertices[ind, 2])/np.pi]
		if face_features[ki,:,0].min() < -0.25 and face_features[ki,:,0].max() > 0.25:
			face_features[ki,:,0][face_features[ki,:,0] < 0] = face_features[ki,:,0][face_features[ki,:,0] < 0] + 1
	face_features = (0.5 + face_features)#*0.98 + 0.01
	return face_features

