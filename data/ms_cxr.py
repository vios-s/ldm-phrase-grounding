from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils import data

from .utils import input_transformations, checkCoord


class MSCXR(data.Dataset):
    def __init__(self, transform_name=None, crop_size=448) -> None:
        base_dir = Path('/remote/rds/groups/idcom_imaging')
        images_dir = 'data/MIMIC_CXR/images/'
        ms_dir = base_dir / 'data/MS_CXR/physionet.org/files/ms-cxr/0.1/MS_CXR_Local_Alignment_v1.0.0.csv'

        df = pd.read_csv(ms_dir)

        # merge boxes in the same img that correspond to the same text
        temp_df = df.groupby(["path", "label_text"]).head(1)
        p1 = temp_df.path.values.tolist()
        l1 = temp_df.label_text.values.tolist()
        i1 = temp_df.index.tolist()
        df = df.astype({"x": "object", "y": "object", "w": "object", "h": "object"})
        for idx, fname, lbl_txt in zip(i1, p1, l1):
            r = df[(df.path == fname) & (df.label_text == lbl_txt)]
            if len(r) == 1:
                continue
            x, y, w, h = r.x.values.tolist(), r.y.values.tolist(), r.w.values.tolist(), r.h.values.tolist()
            df.at[idx, "x"], df.at[idx, "y"], df.at[idx, "w"], df.at[idx, "h"] = x, y, w, h
        df.drop_duplicates(subset=["path", "label_text"], inplace=True, keep="first", ignore_index=True)

        self.df = df
        self.base_dir = base_dir
        self.images_dir = images_dir
        self.transforms = input_transformations(transform_name, crop_size=crop_size)
        self.transform_name = transform_name
    
    def get_class_names(self):
        return self.df.category_name.unique()
    
    def __len__(self):
        return len(self.df)
    
    def get_report_sentences(self, idx):
        reports_dir = self.base_dir / 'data/MIMIC_CXR_report_sentences'
        study_id = self.df.path[idx].split('/')[-2]
        report = reports_dir / f"{study_id}.json"

        try:
            with open(report, "r") as f:
                reportData = json.load(f)
        except FileNotFoundError:
            reportData = {"sentences": []}

        return reportData["sentences"]
    
    def __getitem__(self, idx):
        """
        Returns:
            transformed image (C, H, W) --> torch.Tensor
            transformed bounding box(es) in (x, y, w, h) format --> List[List[int]]
            original bounding box(es) coordinates --> List[List[int]]
            text prompt --> str
            original image path --> str
        """

        image_id = self.df.path[idx]
        image_id = self.base_dir / self.images_dir / image_id
        im_w, im_h = self.df.image_width[idx], self.df.image_height[idx]
        prompt = self.df.label_text[idx]
        x, y, w, h = self.df.x[idx], self.df.y[idx], self.df.w[idx], self.df.h[idx]

        if isinstance(x, list):
            original_bbox = [[e1, e2, e3, e4] for e1, e2, e3, e4 in zip(x, y, w, h)]
        else:
            original_bbox =[[x, y, w, h]]

        # sanity check for bbox coordinates
        # for i, bb in enumerate(original_bbox):
        #     x, y, w, h = bb
        #     x, x_end, y, y_end = checkCoord(x, im_w), checkCoord(x+w, im_w), checkCoord(y, im_h), checkCoord(y+h, im_h)
        #     original_bbox[i] = [x, y, x_end - x, y_end - y]

        if self.transforms is not None:
            bbox = torch.zeros((1, im_h, im_w), dtype=torch.int64)
            for v, bb in enumerate(original_bbox, 1):
                x, y, w, h = bb
                bbox[:, y:y+h, x:x+w] = v * 5  # multiply by a larger number to avoid interpolation issues!
        img_dict = self.transforms({'image': image_id, "bbox": bbox, "mask": torch.ones((im_h, im_w))})  # mask only for 'medrpg'
        img = img_dict["image"]
        bbox = img_dict["bbox"]
        if self.transforms is not None:
            final_bbox, remove_idx = [], 0
            for v in range(1, len(original_bbox) + 1):
                _, y_inds, x_inds = torch.nonzero(bbox == v * 5, as_tuple=True)
                try:
                    temp_bbox = [x_inds[0].item(), y_inds[0].item(), (x_inds[-1] - x_inds[0]).item(), (y_inds[-1] - y_inds[0]).item()]
                    final_bbox.append(temp_bbox)
                except IndexError:
                    # avoid these boxes!
                    print(f"Index {idx} has a [-1,] * 4 bbox", flush=True)
                    original_bbox.pop(v - 1 - remove_idx)
                    remove_idx += 1
                    # final_bbox.append([-1,] * 4)
                    continue
        else:
            final_bbox = bbox

        if self.transform_name == "medrpg":
            return img, final_bbox, original_bbox, img_dict["mask"], prompt, str(image_id), self.df.category_name[idx]
        return img, final_bbox, original_bbox, prompt, str(image_id), self.df.category_name[idx]