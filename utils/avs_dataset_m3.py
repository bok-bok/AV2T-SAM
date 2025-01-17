import os
import pickle
import sys

import clip
import cv2
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.config_m3 import cfg

sys.path.append("CLAP")
import copy
import random

from msclap import CLAP


def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_image_in_cv2_to_Tensor(path, mode='RGB', transform=None):
    image = cv2.imread(path)
    image = cv2.resize(image, (1024, 1024))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach()# [5, 1, 96, 64]
    return audio_log_mel

    



class MS3Dataset_SAM(Dataset):
    """Dataset for multiple sound source segmentation"""
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


    def __init__(self, split):
        super(MS3Dataset_SAM, self).__init__()
        self.device = "cuda"

        image_size = 224
        self.split = split

            
        self.mask_num = 5
        df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform = ResizeLongestSide(1024)
        self.image_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), interpolation=3, antialias=None), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        

        # load CLAP and CLIP models
        pretrained_path ="pretrained/CLAP_weights_2023.pth" 
        self.clap_model = CLAP(model_fp = pretrained_path, version = '2023', use_cuda=True)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device='cuda')
        self.clip_model.float()


  

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        return x

    def _get_clap_embeddings(self, path, clap_embedding_path):
        # check if the embeddings are already saved
        clap_embedding_base_path = "/".join(clap_embedding_path.split("/")[:-1])
        if not os.path.exists(clap_embedding_base_path):
            os.makedirs(clap_embedding_base_path)
        
        if not os.path.exists(clap_embedding_path):
            with torch.no_grad():
                audio_embeddings = self.clap_model.get_audio_embeddings_single(path)
            audio_embeddings = audio_embeddings.detach().cpu()
            # save the embeddings
            with open(clap_embedding_path, 'wb') as fw:
                pickle.dump(audio_embeddings, fw)
        # load the embeddings
        else:
            with open(clap_embedding_path, 'rb') as fr:
                audio_embeddings = pickle.load(fr)
        return audio_embeddings


    def _get_clip_embeddings(self, path, clip_embedding_path):
        # make directory if not exist 
        clip_embedding_base_path = "/".join(clip_embedding_path.split("/")[:-1])
        if not os.path.exists(clip_embedding_base_path):
            os.makedirs(clip_embedding_base_path)

        # check if the embeddings are already saved
        if not os.path.exists(clip_embedding_path):
            image = self.clip_preprocess(Image.open(path)).unsqueeze(0).to('cuda')
            with torch.no_grad():
                image_features:torch.Tensor = self.clip_model.encode_image(image)
            image_features = image_features.squeeze().detach().cpu()
            # save the embeddings
            with open(clip_embedding_path, 'wb') as fw:
                pickle.dump(image_features, fw)
            
        else:
            with open(clip_embedding_path, 'rb') as fr:
                image_features = pickle.load(fr)
        return image_features
        
        

    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        # video_name = df_one_video[0]
        video_name = df_one_video.iloc[0]
        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, video_name)
        # audio_lm_path = os.path.join(cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, video_name + '.pkl')
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, video_name)
        audio_wav_path = os.path.join(cfg.DATA.DIR_AUDIO_WAV, self.split, video_name + '.wav')
        
        
        clap_embedding_path = os.path.join(cfg.DATA.DIR_CLAP, self.split, video_name + '.pkl')
        clip_embedding_path_base = os.path.join(cfg.DATA.DIR_CLIP, self.split, video_name)

        # audio_log_mel = load_audio_lm(audio_lm_path)
        sam_imgs, beit_imgs,  masks = [], [], []
        clip_embeddings  = []
        original_size_list = []
        

            


        clap_embeddings = self._get_clap_embeddings(audio_wav_path, clap_embedding_path)
            




        for img_id in range(1, 6):
            # img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id)), transform=self.img_transform)
            img_path = os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id))
            clip_embedding_path = clip_embedding_path_base + "_%d.pkl"%img_id


            # load image and clip embeddings
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            clip_embedding = self._get_clip_embeddings(img_path, clip_embedding_path)

            original_size_list.append(image.shape[:2])


            # preprocess
            image_evf = self.image_preprocessor(image)

            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]
            image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

            # append images
            sam_imgs.append(image)
            beit_imgs.append(image_evf)
            clip_embeddings.append(clip_embedding)

        # load masks
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='P')
            masks.append(mask)
        
                
        
        sam_imgs_tensor = torch.stack(sam_imgs, dim=0)
        beit_imgs_tensor = torch.stack(beit_imgs, dim=0)

        masks_tensor = torch.stack(masks, dim=0)
        clip_embeddings = torch.stack(clip_embeddings, dim=0)
        
        return sam_imgs_tensor, beit_imgs_tensor,clip_embeddings, clap_embeddings, masks_tensor, original_size_list, video_name

    def __len__(self):
        return len(self.df_split)