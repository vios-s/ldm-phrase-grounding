from typing import Any, List, Tuple
from copy import deepcopy
import torch
from torchvision.ops import box_iou
from sklearn.metrics import f1_score, jaccard_score
# from torcheval.metrics import BinaryBinnedAUROC
from torcheval.metrics.functional import binary_binned_auroc
from monai.transforms import KeepLargestConnectedComponent

from .utils import nanvar, largest_connected_component, get_bb_from_largest_component


lcc = KeepLargestConnectedComponent()


def CNR(sim_map: torch.Tensor, bboxes: List[Tuple], non_absolute: bool = False):
    '''
    Contrast-to-Noise Ratio per given bounding box
    '''
    sim_map = deepcopy(sim_map)
    cnrs = []
    
    for bbox in bboxes:
        x, y, width, height = bbox
        area_in = sim_map[y:y+height, x:x+width]
        mu_in, var_in = torch.nanmean(area_in), nanvar(area_in)
        sim_map[y:y+height, x:x+width] = float("NaN")
        mu_out, var_out = torch.nanmean(sim_map), nanvar(sim_map)
        if non_absolute:
            means_diff = mu_in - mu_out
        else:
            means_diff = torch.abs(mu_in - mu_out)
        cnr = means_diff / torch.sqrt(var_in + var_out)
        cnrs.append(torch.nan_to_num(cnr).item())
    
    return sum(cnrs) / len(cnrs)


def mIoU_old(sim_map: torch.Tensor, gt_boxes: List[List], thresholds: List = [.1, .2, .3, .4, .5]):
    # Old version of mIoU --> extracts a bbox via largest connected component

    iou_per_bb = []
    for gt_box in gt_boxes:
        gt_box = torch.tensor(gt_box)[None, :]
        gt_box[:, 2] += gt_box[:, 0]
        gt_box[:, 3] += gt_box[:, 1]
    
        ious = torch.empty((len(thresholds),))
        for i, threshold in enumerate(thresholds):
            # get predicted bounding box from similarity map
            sim_map_bin = sim_map > threshold
            largest_comp = lcc(sim_map_bin[None, ...]).squeeze(0)
            try:
                y_inds, x_inds = torch.nonzero(largest_comp, as_tuple=True)
                pred_box = [x_inds[0].item(), y_inds[0].item(), x_inds[-1].item(), y_inds[-1].item()]
            except IndexError:
                ious[i] = 0
                continue

            # previous implementation
            # largest_comp = largest_connected_component(sim_map_bin)
            # pred_box = get_bb_from_largest_component(largest_comp)

            pred_box = torch.tensor(pred_box)[None, :]
            # pred_box[:, 2] += pred_box[:, 0]
            # pred_box[:, 3] += pred_box[:, 1]
            # IoU @ threshold
            ious[i] = box_iou(pred_box, gt_box).squeeze().item()
        
        iou_per_bb.append(ious.mean().item())

    return sum(iou_per_bb) / len(iou_per_bb)


def mIoU(sim_map: torch.Tensor, gt_boxes: List[List], thresholds: List = [.1, .2, .3, .4, .5]):
    # iou_per_bb = []
    trg = torch.zeros_like(sim_map, dtype=torch.bool)
    for gt_box in gt_boxes:
        x, y, w, h = gt_box
        trg[y:y+h, x:x+w] = True
    
    ious = torch.empty((len(thresholds),))
    for i, threshold in enumerate(thresholds):
        bin_mask = sim_map > threshold
        intersection = torch.logical_and(bin_mask, trg)
        union = torch.logical_or(bin_mask, trg)
        # IoU @ threshold
        ious[i] = intersection.sum().item() / union.sum().item()
    
    # iou_per_bb.append(ious.mean().item())
    # return sum(iou_per_bb) / len(iou_per_bb)
    return ious.mean().item()


def mIoU_scikit(sim_map: torch.Tensor, gt_boxes: List[List], thresholds: List = [.1, .2, .3, .4, .5]):
    iou_per_bb = []
    for bb in gt_boxes:
        x, y, w, h = bb
        trg = torch.zeros_like(sim_map, dtype=torch.bool)
        trg[y:y+h, x:x+w] = True
    
        ious = torch.empty((len(thresholds),))
        for i, threshold in enumerate(thresholds):
            bin_mask = sim_map > threshold
            ious[i] = jaccard_score(trg.flatten().numpy(), bin_mask.flatten().numpy())
        
        iou_per_bb.append(ious.mean().item())

    return sum(iou_per_bb) / len(iou_per_bb)


class AUC_ROC:
    def __init__(self):
        # self.metric = BinaryBinnedAUROC(threshold=5)
        pass
    
    def __call__(self, sim_map: torch.Tensor, gt_boxes: List[List]):
        trg = torch.zeros_like(sim_map, dtype=torch.bool)
        for bb in gt_boxes:
            x, y, w, h = bb
            trg[y:y+h, x:x+w] = True
        
        # self.metric.update(sim_map.flatten(), trg.long().flatten())
        # return self.metric.compute()[0].item()
        return binary_binned_auroc(sim_map.flatten(), trg.long().flatten(), threshold=5)[0].item()
