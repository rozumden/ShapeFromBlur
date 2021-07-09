import numpy as np
import math 

from skimage.draw import line_aa
from skimage import measure
import skimage.transform
from scipy import signal
from skimage.measure import label, regionprops
import skimage.metrics as metrics
import scipy.misc
import cv2
import yaml

def imread(name):
	img = cv2.imread(name,cv2.IMREAD_UNCHANGED)
	return img[:,:,[2,1,0]]/255

def imwrite(im, name):
	im[im<0]=0
	im[im>1]=1
	cv2.imwrite(name, im[:,:,[2,1,0]]*255)

def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config

def fmo_detect(I,B):
	## simulate FMO detector -> find approximate location of FMO
	dI = (np.sum(np.abs(I-B),2) > 0.05).astype(float)
	labeled = label(dI)
	regions = regionprops(labeled)
	ind = -1
	maxsol = 0
	for ki in range(len(regions)):
		if regions[ki].area > 100 and regions[ki].area < 0.01*np.prod(dI.shape):
			if regions[ki].solidity > maxsol:
				ind = ki
				maxsol = regions[ki].solidity
	if ind == -1:
		return [], 0
	
	bbox = np.array(regions[ind].bbox).astype(int)
	return bbox, regions[ind].minor_axis_length

def fmo_detect_maxarea(I,B):
	## simulate FMO detector -> find approximate location of FMO
	dI = (np.sum(np.abs(I-B),2) > 0.05).astype(float)
	labeled = label(dI)
	regions = regionprops(labeled)
	ind = -1
	maxarea = 0
	for ki in range(len(regions)):
		if regions[ki].area > maxarea:
			ind = ki
			maxarea = regions[ki].area
	if ind == -1:
		return [], 0
	bbox = np.array(regions[ind].bbox).astype(int)
	return bbox, regions[ind].minor_axis_length

def fmo_detect_hs(gt_hs,B):
	dI = (np.sum((np.sum(np.abs(gt_hs-B[:,:,:,None]),2) > 0.1),2) > 0.5).astype(float)
	labeled = label(dI)
	regions = regionprops(labeled)
	ind = -1
	maxarea = 0
	for ki in range(len(regions)):
		if regions[ki].area > maxarea:
			ind = ki
			maxarea = regions[ki].area
	if ind == -1:
		return [], 0
	bbox = np.array(regions[ind].bbox).astype(int)
	return bbox, regions[ind].minor_axis_length

def bbox_detect_hs(gt_hs,B):
	dI = (np.sum(np.abs(gt_hs-B),2) > 0.1).astype(float)
	labeled = label(dI)
	regions = regionprops(labeled)
	ind = -1
	maxarea = 0
	for ki in range(len(regions)):
		if regions[ki].area > maxarea:
			ind = ki
			maxarea = regions[ki].area
	if ind == -1:
		return []
	bbox = np.array(regions[ind].bbox).astype(int)
	return bbox

def fmo_model(B,H,F,M):
	if len(H.shape) == 2:
		H = H[:,:,np.newaxis]
		F = F[:,:,:,np.newaxis]
	elif len(F.shape) == 3:
		F = np.repeat(F[:,:,:,np.newaxis],H.shape[2],3)
	HM3 = np.zeros(B.shape)
	HF = np.zeros(B.shape)
	for hi in range(H.shape[2]):
		M1 = M
		if len(M.shape) > 2:
			M1 = M[:, :, hi]
		M3 = np.repeat(M1[:, :, np.newaxis], 3, axis=2)
		HM = signal.fftconvolve(H[:,:,hi], M1, mode='same')
		HM3 += np.repeat(HM[:, :, np.newaxis], 3, axis=2)
		F3 = F[:,:,:,hi]
		for kk in range(3):
			HF[:,:,kk] += signal.fftconvolve(H[:,:,hi], F3[:,:,kk], mode='same')
	I = B*(1-HM3) + HF
	return I

def boundingBox(img, pads=None):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    if pads is not None:
    	rmin = max(rmin - pads[0], 0)
    	rmax = min(rmax + pads[0], img.shape[0])
    	cmin = max(cmin - pads[1], 0)
    	cmax = min(cmax + pads[1], img.shape[1])
    return rmin, rmax, cmin, cmax
    

def extend_bbox(bbox,ext,aspect_ratio,shp):
	height, width = bbox[2] - bbox[0], bbox[3] - bbox[1]
			
	h2 = height + ext

	h2 = int(np.ceil(np.ceil(h2 / aspect_ratio) * aspect_ratio))
	w2 = int(h2 / aspect_ratio)

	wdiff = w2 - width
	wdiff2 = int(np.round(wdiff/2))
	hdiff = h2 - height
	hdiff2 = int(np.round(hdiff/2))

	bbox[0] -= hdiff2
	bbox[2] += hdiff-hdiff2
	bbox[1] -= wdiff2
	bbox[3] += wdiff-wdiff2
	bbox[bbox < 0] = 0
	bbox[2] = np.min([bbox[2], shp[0]-1])
	bbox[3] = np.min([bbox[3], shp[1]-1])
	return bbox

def extend_bbox_uniform(bbox,ext,shp):
	bbox[0] -= ext
	bbox[2] += ext
	bbox[1] -= ext
	bbox[3] += ext
	bbox[bbox < 0] = 0
	bbox[2] = np.min([bbox[2], shp[0]-1])
	bbox[3] = np.min([bbox[3], shp[1]-1])
	return bbox

def extend_bbox_nonuniform(bbox,ext,shp):
	bbox[0] -= ext[0]
	bbox[2] += ext[0]
	bbox[1] -= ext[1]
	bbox[3] += ext[1]
	bbox[bbox < 0] = 0
	bbox[2] = np.min([bbox[2], shp[0]-1])
	bbox[3] = np.min([bbox[3], shp[1]-1])
	return bbox

def bbox_fmo(bbox,gt_hs,B):
	gt_hs_crop = crop_only(gt_hs,bbox)
	B_crop = crop_only(B,bbox)
	bbox_crop,rad = fmo_detect_hs(gt_hs_crop,B_crop)
	bbox_new = bbox_crop
	if len(bbox_new) > 0:
		bbox_new[:2] += bbox[:2]
		bbox_new[2:] += bbox[:2]
	else:
		bbox_new = bbox
	return bbox_new

def rgba2hs(rgba, bgr):
	return rgba[:,:,:3]*rgba[:,:,3:] + bgr[:,:,:,None]*(1-rgba[:,:,3:])

def rgba2rgb(rgba):
	return rgba[:,:,:3]*rgba[:,:,3:] + 1*(1-rgba[:,:,3:])

def crop_resize(Is, bbox, res):
	if Is is None:
		return None
	rev_axis = False
	if len(Is.shape) == 3:
		rev_axis = True
		Is = Is[:,:,:,np.newaxis]
	imr = np.zeros((res[1], res[0], Is.shape[2], Is.shape[3]))
	for kk in range(Is.shape[3]):
		im = Is[bbox[0]:bbox[2], bbox[1]:bbox[3], :, kk]
		imr[:,:,:,kk] = cv2.resize(im, res, interpolation = cv2.INTER_CUBIC)
	if rev_axis:
		imr = imr[:,:,:,0]
	return imr

def crop_only(Is, bbox):
	if Is is None:
		return None
	return Is[bbox[0]:bbox[2], bbox[1]:bbox[3]]

def rev_crop_resize(inp, bbox, I):
	est_hs = np.tile(I.copy()[:,:,:,np.newaxis],(1,1,1,inp.shape[3]))
	for hsk in range(inp.shape[3]):
		est_hs[bbox[0]:bbox[2], bbox[1]:bbox[3],:,hsk] = cv2.resize(inp[:,:,:,hsk], (bbox[3]-bbox[1],bbox[2]-bbox[0]), interpolation = cv2.INTER_CUBIC)
	return est_hs

