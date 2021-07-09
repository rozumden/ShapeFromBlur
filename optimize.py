import argparse
import os
import shutil
import torch
import time
from utils import *
from shapefromblur import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--im", required=False, default="examples/vol_im.png")
    parser.add_argument("--bgr", required=False, default="examples/vol_bgr.png")
    parser.add_argument("--config", required=False, default="configs.yaml")
    parser.add_argument("--subframes", required=False, default=8)
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(config)
    if not os.path.exists(config["write_results_folder"]):
        os.makedirs(config["write_results_folder"])
        shutil.copyfile(os.path.join('prototypes','model.mtl'),os.path.join(config["write_results_folder"],'model.mtl'))

    sfb = ShapeFromBlur(config, device)

    I = imread(args.im)
    B = imread(args.bgr)
    bbox_tight, radius = fmo_detect_maxarea(I,B)
    bbox_tight = extend_bbox_nonuniform(bbox_tight,[10, 10],I.shape[:2])

    t0 = time.time()
    best_model = sfb.apply(I, B, bbox_tight, args.subframes, radius, None)
    print('{:4d} epochs took {:.2f} seconds, best model loss {:.4f}'.format(config["iterations"], (time.time() - t0)/1, best_model["value"]))

    est = rev_crop_resize(best_model["renders"][0,0].transpose(2,3,1,0), sfb.bbox, np.zeros((I.shape[0],I.shape[1],4)))
    est_hs_tight = crop_only(rgba2hs(est, B),bbox_tight)
   
    for ki in range(args.subframes):
        imwrite(est_hs_tight[:,:,:,ki],os.path.join(config["write_results_folder"],'est{}.png'.format(ki)))
    imwrite(crop_only(B,bbox_tight),os.path.join(config["write_results_folder"],'input_bgr.png'))
    imwrite(crop_only(I,bbox_tight),os.path.join(config["write_results_folder"],'input_im.png'))

if __name__ == "__main__":
    main()