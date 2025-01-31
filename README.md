# Zero-Shot Medical Phrase Grounding with Off-the-shelf Diffusion Models
Official repository for [Zero-Shot Medical Phrase Grounding with Off-the-shelf Diffusion Models](https://arxiv.org/abs/2404.12920) accepted at IEEE JBHI (Special Issue on Foundation Models in Medical Imaging).

## Getting started
To reproduce all experiments, the following steps need to be completed first:

### Data
Our work is based on [MS-CXR](https://physionet.org/content/ms-cxr/1.1.0/), which is a subset of the large-scale [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) dataset. Please note that only credentialed PhysioNet users can access both datasets.

### Python environment
Create a virtual environment using the provided `requirements.txt` file

```python
# via pip
pip install -r requirements.txt

# via Conda
conda create --name <your_env_name> --file requirements.txt
```

### LDM weights
Instructions on how to download weights for the LDM pre-trained on MIMIC-CXR can be found in [1] (see below). The downloaded checkpoints are expected to be in a directory called `models/`

### Baseline models
Code for instantiating both [BioViL](https://arxiv.org/abs/2204.09817) and [BioViL-T](https://arxiv.org/abs/2301.04558) models is provided in the [HI-ML Multimodal Toolbox](https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal) repository. You can either install the toolbox via pip or clone the repository in `health_multimodal` directory -- see [2] below.

## LDM evaluation
To perform phrase grounding with the pre-trained LDM, you can run the following script:

```bash
python3 eval_ldm.py
```

## BioViL(-T) evaluation
To perform phrase grounding with either BioViL or BioViL-T (this can be controlled through the `model-name` argument), you can run the following script:

```bash
python3 eval_biovil_t.py --model-name biovil_t
```

## Acknowledgements
1. https://github.com/Project-MONAI/GenerativeModels/tree/main/model-zoo/models/cxr_image_synthesis_latent_diffusion_model
   * Links for pre-trained model weights can be found in the `large_files.yml` file.
2. https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal/src/health_multimodal
3. https://github.com/Warvito/generative_chestxray
