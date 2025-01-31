from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from monai.utils import optional_import
from torchvision.transforms import Resize
from scipy import ndimage
from skimage.filters import threshold_multiotsu
from models_local import attention, get_models

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")
DISABLE = int(os.getenv("TQDM_DISABLE", 0))


def get_mask_otsu(heatmap: torch.Tensor, n_classes: int = 2):
    device = heatmap.device
    thresholds = threshold_multiotsu(heatmap.cpu().numpy(), classes = n_classes)
    thresholds = torch.from_numpy(thresholds)
    mask_binary = heatmap[..., None] > thresholds[None, None, :]
    mask_binary = mask_binary.any(dim=-1)
    return mask_binary.to(device=device, dtype=torch.float32)


def normalise_to_01(heatmap):
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap


def get_heatmap_from_attention(
        diffusion,
        token_init_pos,
        token_final_pos,
        steps,
        final_size=64
    ):
    cross_attention_layers = attention.find_cross_attention_layers(diffusion)
    attention_maps_per_layer = []
    for idx, cross_attention_layer in enumerate(cross_attention_layers):
        if idx not in [3, 4, 6, 7]:
            continue
        # stack all timesteps in the last dimension
        attention_map = torch.stack(cross_attention_layer.attention_scores_list, -1)
        # get spatial attention maps
        shape_init = attention_map.shape
        spatial_dim = int(np.sqrt(shape_init[1]))

        attention_map_spatial = attention_map.reshape(shape_init[0], spatial_dim, spatial_dim, shape_init[2], shape_init[3])
        attention_map_spatial = attention_map_spatial[1]  # cross-attention maps from conditional input
        
        # select the tokens (and timesteps) of interest
        attention_map_spatial = attention_map_spatial[..., token_init_pos:token_final_pos, steps].mean(-2)
        # normalise (0,1) in spatial dimensions
        max_ = attention_map_spatial.max(0)[0].max(0)[0]
        min_ = attention_map_spatial.min(0)[0].min(0)[0]
        attention_map_spatial = (attention_map_spatial - min_) / (max_- min_)
        # move last axis to first position (batch dimension)
        attention_map_spatial = attention_map_spatial.permute(2,0,1)
        # resize to fit with the latent dimension
        attention_map_spatial = Resize(final_size)(attention_map_spatial)
        attention_maps_per_layer.append(attention_map_spatial)
    attention_map_spatial = torch.stack(attention_maps_per_layer)  # shape: [layers, timesteps, latent_dim, latent_dim]
    # average over layers and timesteps
    heatmap = attention_map_spatial.mean([0,1])
    heatmap = torch.tensor(
        ndimage.gaussian_filter(heatmap.numpy(), sigma=(2.5, 2.5), order=0)
    )
    return heatmap


class Sampler:
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def sampling_fn(
        self,
        image: torch.Tensor,
        prompt: str,
        autoencoder_model: nn.Module,
        diffusion_model: nn.Module,
        scheduler: nn.Module,
        text_encoder: nn.Module,
        tokenizer: nn.Module,
        steps: range,
        guidance_scale: float = 7.0,
        scale_factor: float = 0.3,
        cls_name: str = None
    ) -> torch.Tensor:
        prompts = ['', prompt]  # uncond + cond
        prompt_embeds = get_models.get_prompt_embeds(prompts, tokenizer, text_encoder).to(device=image.device, dtype=torch.float32)

        image_encoding_ = autoencoder_model.encode(image)[0] * scale_factor
        image_encoding = image_encoding_

        attention.modify_cross_attention_layers(diffusion_model)

        timesteps_inv = scheduler.timesteps.flip(0)  # [0, end]

        for t in tqdm(timesteps_inv, disable=DISABLE):
            noise_input = torch.cat([image_encoding] * 2)
            model_output = diffusion_model(
                noise_input, timesteps=torch.Tensor((t,)).to(image.device).long(), context=prompt_embeds
            )
            model_output_uncond, model_output_text = model_output.chunk(2)
            noise_pred = model_output_uncond + guidance_scale * (model_output_text - model_output_uncond)
            image_encoding, _ = scheduler.reversed_step(noise_pred, t, image_encoding)
        
        text_inputs = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        token_init_pos, token_final_pos = 1, len(text_inputs.input_ids.squeeze()) - 1

        # this block of code corresponds to the attribution experiments (see Table III in our paper)
        if cls_name is not None:
            cls_tokens = tokenizer(cls_name, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            cls_tokens = cls_tokens.input_ids.squeeze()[1:-1]
            cls_tokens = torch.nonzero(text_inputs.input_ids.squeeze()[:, None] == cls_tokens, as_tuple=True)[0]
            if cls_tokens.numel() == 0:
                return {'heatmap': torch.full((image.shape[-1],) * 2, torch.nan)}
            token_init_pos, token_final_pos = cls_tokens[0].item(), cls_tokens[-1].item()
            if token_init_pos == token_final_pos:
                token_final_pos += 1

        heatmap = get_heatmap_from_attention(
            diffusion_model, 
            token_init_pos, 
            token_final_pos,
            steps,
            final_size=image.shape[-1],
        )

        # post-processing
        heatmap = normalise_to_01(heatmap)
        bin_mask = get_mask_otsu(heatmap, n_classes=2)  # thresholding
        heatmap = heatmap * bin_mask

        return {'heatmap': heatmap}


        # with autocast():
        #     sample = autoencoder_model.decode_stage_2_outputs(noise / scale_factor)
        # return sample