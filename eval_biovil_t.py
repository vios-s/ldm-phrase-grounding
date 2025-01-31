import logging
import argparse
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
# local
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.vlp import ImageTextInferenceEngine
from data import MSCXR
from metrics import *


parser = argparse.ArgumentParser(description="Evaluate BioVIL(-T) on phrase grounding")
parser.add_argument("--model-name", type=str, default="biovil_t", choices=["biovil", "biovil_t"])
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

# models
logging.info(f"Model: {args.model_name}")
if args.model_name == "biovil":
    bert_encoder_type = BertEncoderType.CXR_BERT
    image_model_type = ImageModelType.BIOVIL
    crop_size = 480
else:
    bert_encoder_type = BertEncoderType.BIOVIL_T_BERT
    image_model_type = ImageModelType.BIOVIL_T
    crop_size = 448

text_inference = get_bert_inference(bert_encoder_type)
image_inference = get_image_inference(image_model_type)
image_text_inference = ImageTextInferenceEngine(
    image_inference_engine=image_inference,
    text_inference_engine=text_inference,
)
image_text_inference.to(device)
# data
ds = MSCXR(transform_name="biovil_t", crop_size=crop_size)
# metrics
auc_roc = AUC_ROC()
cnrs, mious = {k: [] for k in ds.get_class_names()}, {k: [] for k in ds.get_class_names()}
aucrocs = deepcopy(cnrs)
nonabs_cnrs = deepcopy(cnrs)
for idx in range(len(ds)):
    img, bbox, original_bbox, prompt, image_id, cls_name = ds[idx]
    
    if len(bbox) < 1:
        print(f"Skipped sample {idx}", flush=True)
        continue
    
    sim_map = image_text_inference.get_similarity_map_from_raw_data(Path(image_id), prompt)
    sim_map = torch.from_numpy(sim_map)
    sim_map = sim_map.clamp(min=0)  # note that sim_map = (sim_map + 1) / 2  yields worse results!

    cnr = CNR(sim_map, original_bbox)
    miou = mIoU(sim_map, original_bbox)
    aucroc = auc_roc(sim_map, original_bbox)
    cnrs[cls_name].append(cnr)
    mious[cls_name].append(miou)
    nonabs_cnrs[cls_name].append(CNR(sim_map, original_bbox, non_absolute=True))
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