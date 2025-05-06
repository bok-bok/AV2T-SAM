# Audio Visual Segmentation Through Text Embeddings
---
This is the official repository of the paper "Audio Visual Segmentation Through Text Embeddings"

## Requirements
---
### Environment
```shell
conda create -n avt python=3.9.20
conda activate avt

pip install -r requirements.txt
```

### Data preparation 
#### 1. Download the datasets 

- Follow the guidance of [https://github.com/OpenNLPLab/AVSBench](https://github.com/OpenNLPLab/AVSBench) to download the AVSBench dataset.
#### 2. dataset location configuration
- Modify path root variables in utils/config_m3.py and utils/config_s4.py. 

### pre-trained backbones 
- Place all weights from [here](https://huggingface.co/Kyungbok/AV2T-SAM/tree/main) to pretrained directory. 
---

### Training 
#### S4
```shell
python train_avs.py --evf_version evf_sam2 --projector_type mul --use_adapter --dataset s4 --batch_size 8
```
#### M3
```shell
python train_avs.py --evf_version evf_sam2 --projector_type mul --use_adapter --dataset m3 --batch_size 8
```

### Testing 
Replace --name and --weight_path to appropriate name andm weight_path. 
#### S4
```shell
python test_avs.py --dataset s4 --name name --evf_version evf_sam2 --projector_type mul --use_adapter --adapter_type mul --weight_path weight
```
#### M3
```shell
python test_avs.py --dataset m3 --name name --evf_version evf_sam2 --projector_type mul --use_adapter --adapter_type mul --weight_path weight
```