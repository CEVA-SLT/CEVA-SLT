## Implementation
- The implementation for the *prior path*, the *posterior path*, the proposed *Language-guided Prefix Enhancement (LGPE)* module, and the *Dynamic Representation Enhancement (DRE)* strategy is provided in `./modeling/translation.py`

- The implementation for the proposed *Refined Contrastive Variational Alignment (RCVA)* module is provided in `./modeling/gaussian_net_5_t.py` and `./modeling/contrastive_loss.py`

### Prerequisites 

```sh
conda env create -f environment.yml
conda activate slt
```

### Data preparation

Please refer to the implementation of MMTLB for preparing the data and pretrained models, as CEVA-SLT focuses on the SLT training stage. Specifically, the required processed data and pretrained models include:

- Pre-extracted visual features for PHOENIX-2014T and CSL-Daily are obtained following the MMTLB implementation. Please download and place them under `./experiment`.

- Pre-trained Visual Embedding (trained on the S2G task) and MBart modules (trained on the G2T task) are adopted following the MMTLB implementation. Please download the corresponding directories and place them under `./pretrained_models`.

- Qwen2-1.5B is adopted from the official pretrained model released by Qwen. Please download the model and place it under `./models`.

> Note that the path is configured in the `*.yaml` file and can be modified according to your local environment.

### Train and Evaluate

**Train**

```
dataset=phoenix-2014t #phoenix14t / csl-daily
python -m torch.distributed.launch \
--nproc_per_node 1 \
--use_env training.py \
--config experiments/configs/SingleStream/${dataset}_vs2t.yaml

```

**Evaluate**

Upon finishing training, your can evaluate the model with:

```
dataset=phoenix-2014t #phoenix14t / csl-daily
python -m torch.distributed.launch \
--nproc_per_node 1 \
--use_env prediction.py  \
--config experiments/configs/SingleStream/${dataset}_vs2t.yaml
```

