from easydict import EasyDict as edict

"""
default config
"""
cfg = edict()
###############################
# TRAIN
cfg.TRAIN = edict()

cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "../../pretrained_backbones/vggish-10086976.pth"
cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = False
cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "../../pretrained_backbones/vggish_pca_params-970ea276.pth"
cfg.TRAIN.FREEZE_VISUAL_EXTRACTOR = True

cfg.TRAIN.FINE_TUNE_SSSS = False
cfg.TRAIN.PRETRAINED_S4_aAVS_WO_TPAVI_PATH = "../avs_s4/train_logs/checkpoints/checkpoint_xxx.pth.tar"
cfg.TRAIN.PRETRAINED_S4_AVS_WITH_TPAVI_PATH = "../avs_s4/train_logs/checkpoints/checkpoint_xxx.pth.tar"

###############################
# DATA
path_root = '/projects/bcza/kb7180'
cfg.DATA = edict()
cfg.DATA.ANNO_CSV = f"{path_root}/avsbench_data/Single-source/s4_meta_data.csv"
cfg.DATA.DIR_IMG = f"{path_root}/avsbench_data/Single-source/s4_data/visual_frames"
cfg.DATA.DIR_AUDIO_LOG_MEL = f"{path_root}/avsbench_data/Single-source/s4_data/audio_log_mel"
cfg.DATA.DIR_AUDIO_WAV = f"{path_root}/avsbench_data/Single-source/s4_data/audio_wav"
cfg.DATA.DIR_MASK = f"{path_root}/avsbench_data/Single-source/s4_data/gt_masks"
cfg.DATA.DIR_CLAP = f"{path_root}/avsbench_data/Single-source/s4_data/clap_embeddings"
cfg.DATA.DIR_CLIP = f"{path_root}/avsbench_data/Single-source/s4_data/clip_embeddings"
###############################


