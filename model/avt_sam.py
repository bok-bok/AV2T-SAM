import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoTokenizer

from model.audio_prompt_generator import AudioPromptGenerator
from model.evf_sam import EvfSamModel
from model.evf_sam2 import EvfSam2Model
from model.projector import Projector


class AVTSAM(nn.Module):
    def __init__(self, args):
        super(AVTSAM, self).__init__()
        self.args = args
        self.depth = 32

        self.tokenizer, self.model = self.init_model()
        self.common_dim = 512
        self.visual_dim = 256
        self.projector = Projector(
            args.projector_type,
            audio_embedding_dim=1024,
            vision_embedding_dim=512,
            common_dim=self.common_dim,
        )

        if args.use_adapter:
            self.audio_prompt_generator = AudioPromptGenerator(
                depth=self.depth, aud_in_dim=self.common_dim, aud_hidden_dim=self.common_dim
            )

    def init_model(self):
        if self.args.evf_version == "evf_sam":
            print("Using evf_sam")
            version = "YxZhang/evf-sam-multitask"
            tokenizer = AutoTokenizer.from_pretrained(
                version,
                padding_side="right",
                use_fast=False,
            )
            torch_dtype = torch.float32

            kwargs = {"torch_dtype": torch_dtype}
            model = EvfSamModel.from_pretrained(version, low_cpu_mem_usage=True, **kwargs)

        elif self.args.evf_version == "evf_sam2":
            self.depth = 48
            version = "YxZhang/evf-sam2-multitask"

            tokenizer = AutoTokenizer.from_pretrained(
                version,
                padding_side="right",
                use_fast=False,
            )
            torch_dtype = torch.float32
            kwargs = {"torch_dtype": torch_dtype}

            model = EvfSam2Model.from_pretrained(version, low_cpu_mem_usage=False, args=self.args, **kwargs)

        model = model.cuda()

        return tokenizer, model

    def forward(self, images_sam, images_beit, clip_embeddings, clap_embeddings, original_size_lists):

        # compute text embeddings
        encoder_features = self.projector.get_features(clap_embeddings, clip_embeddings)

        B, T, C, H, W = images_sam.shape
        Bb, Tb, Cb, Hb, Wb = images_beit.shape
        images_sam = images_sam.reshape(B * T, C, H, W)
        images_beit = images_beit.reshape(Bb * Tb, Cb, Hb, Wb)
        # get prompt features
        if self.args.use_adapter:
            encoder_features = encoder_features.reshape(B * T, -1)
            prompt_features = self.audio_prompt_generator(encoder_features)

        else:
            prompt_features = None

        # get image features
        sam_backbone_output = self.model.get_sam_encoder_features(images_sam, prompt_features)

        # get text embeddings
        text_projected_embeddings = self.projector.get_text_embeddings(encoder_features)
        text_projected_embeddings = text_projected_embeddings.reshape(B * T, 3, 1024)

        # [B * T, 1, 1024]
        beit3_features = self.model.get_beit3_features(images_beit, text_projected_embeddings)

        pred_masks = self.get_pred_masks(
            images_sam,
            sam_backbone_output,
            beit3_features,
            prompt_features,
            original_size_lists,
            dense_prompt_embeddings=None,
        )
        return pred_masks

    def get_pred_masks(
        self,
        images,
        sam_backbone_output,
        beit3_features,
        prompt_features,
        original_size_list,
        dense_prompt_embeddings,
    ):

        pred_mask = self.model.inference_text_embeddings(
            images,
            sam_backbone_output,
            beit3_features,
            prompt_features,
            resize_list=[None],
            original_size_list=original_size_list,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        return pred_mask
