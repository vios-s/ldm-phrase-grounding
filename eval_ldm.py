import logging
import argparse
import os
os.environ["TQDM_DISABLE"] = "1"
from pathlib import Path
from math import floor, ceil
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
# local
from data import MSCXR
from metrics import *
from sampler import Sampler
from models_local import get_models


parser = argparse.ArgumentParser(description="Evaluate LDM on phrase grounding")
parser.add_argument("--num-timesteps", type=int, default=300, help="Number of timesteps for DDIM inversion")
parser.add_argument("--gpu-id", type=int, default=0)
parser.add_argument("--log-steps", type=int, default=50, help="Number of steps for logging calls")
args = parser.parse_args()

# logger
logging.basicConfig(
    filename="output.log",  # TODO: change fname if necessary
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="a",
    force=True
)

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
# data
ds = MSCXR(transform_name="ddpm")
# models
path_this_file = Path(__file__).parent.absolute()
stage1_config_file_path = path_this_file / 'configs/stage1/aekl_v0.yaml'
diffusion_config_file_path = path_this_file / 'configs/ldm/ldm_v0.yaml'
stage1_path = path_this_file / 'models/autoencoder.pth'
diffusion_path = path_this_file / 'models/diffusion_model.pth'
stage1, diffusion, scheduler, tokenizer, text_encoder = get_models.get_modules(
    stage1_config_file_path, stage1_path, diffusion_config_file_path, diffusion_path,
    device=device, num_timesteps=args.num_timesteps
)
sampler = Sampler()
# heuristic for timestep selection
t_min, t_max = args.num_timesteps // 2 - 10 * args.num_timesteps // 100, args.num_timesteps // 2 + 10 * args.num_timesteps // 100
# metrics
auc_roc = AUC_ROC()
cnrs, mious = {k: [] for k in ds.get_class_names()}, {k: [] for k in ds.get_class_names()}
aucrocs = deepcopy(cnrs)
nonabs_cnrs = deepcopy(cnrs)

for idx in range(len(ds)):
    img, bbox, original_bbox, prompt, image_id, cls_name = ds[idx]
    img = img.to(device=device, dtype=torch.float32)

    if len(bbox) < 1:
        print(f"Skipped sample {idx}", flush=True)
        continue

    output_dict = sampler.sampling_fn(
        img.to(device=device, dtype=torch.float32)[None, ...],
        prompt,
        stage1, 
        diffusion, 
        scheduler, 
        text_encoder,
        tokenizer,
        range(t_min, t_max),
        guidance_scale = 0,
        scale_factor = 0.3,
        cls_name = None  # None or cls_name (for attribution experiment)
    )
    sim_map = output_dict["heatmap"]

    # to original image dimensions
    w, h = Image.open(Path(image_id)).size
    smallest_dim = min(w, h)
    target_size = smallest_dim, smallest_dim
    sim_map = F.interpolate(
        sim_map[None, None, ...],
        size=target_size,
        mode="nearest",
        align_corners=None,
    )[0, 0]
    margin_w, margin_h = (w - target_size[0]), (h - target_size[1])
    margins_for_pad = (floor(margin_w / 2), ceil(margin_w / 2), floor(margin_h / 2), ceil(margin_h / 2))
    sim_map = F.pad(sim_map, margins_for_pad, value=float("NaN"))

    # metrics
    cnr = CNR(sim_map, original_bbox)
    cnr_nonabs = CNR(sim_map, original_bbox, non_absolute=True)
    miou = mIoU(sim_map, original_bbox)
    aucroc = auc_roc(sim_map, original_bbox)

    cnrs[cls_name].append(cnr)
    mious[cls_name].append(miou)
    nonabs_cnrs[cls_name].append(cnr_nonabs)
    aucrocs[cls_name].append(aucroc)

    # logging
    if idx % args.log_steps == 0:
        logging.info(f"After {idx + 1} samples - MS-CXR results (len={len(ds)})\nCNR: {[(k, np.mean(v)) for k, v in cnrs.items() if len(v) > 0]}\nmIoU: {[(k, np.mean(v)) for k, v in mious.items() if len(v) > 0]}\nAUC-ROC: {[(k, np.mean(v)) for k, v in aucrocs.items() if len(v) > 0]}")
        logging.info(f"Avg |CNR|: {np.mean([np.mean(v) for v in cnrs.values() if len(v) > 0]) :.4f}")
        logging.info(f"Avg mIoU: {np.mean([np.mean(v) for v in mious.values() if len(v) > 0]) :.4f}")
        logging.info(f"Avg AUC-ROC: {np.mean([np.mean(v) for v in aucrocs.values() if len(v) > 0]) :.4f}")
        logging.info(f"Avg CNR: {np.mean([np.mean(v) for v in nonabs_cnrs.values() if len(v) > 0]) :.4f}")

logging.info(f"MS-CXR results (len={len(ds)})\n|CNR|: {np.mean([np.mean(v) for v in cnrs.values()]) :.4f} +- {np.std([np.mean(v) for v in cnrs.values()]) :.4f}\nmIoU: {np.mean([np.mean(v) for v in mious.values()]) :.4f} +- {np.std([np.mean(v) for v in mious.values()]) :.4f}\nAUC-ROC: {np.mean([np.mean(v) for v in aucrocs.values()]) :.4f} +- {np.std([np.mean(v) for v in aucrocs.values()]) :.4f}\nAvg CNR: {np.mean([np.mean(v) for v in nonabs_cnrs.values() if len(v) > 0]) :.4f} +- {np.std([np.mean(v) for v in nonabs_cnrs.values()]) :.4f}")
logging.info("******CNR results******")
for k, v in cnrs.items():
    logging.info(f"{k}: {np.mean(v) :.4f} +- {np.std(v) :.4f}")

logging.info("******mIoU results******")
for k, v in mious.items():
    logging.info(f"{k}: {np.mean(v) :.4f} +- {np.std(v) :.4f}")

logging.info("******Non-absolute CNR results******")
for k, v in nonabs_cnrs.items():
    logging.info(f"{k}: {np.mean(v) :.4f} +- {np.std(v) :.4f}")

logging.info("******AUC-ROC results******")
for k, v in aucrocs.items():
    logging.info(f"{k}: {np.mean(v) :.4f} +- {np.std(v) :.4f}")