import torch
from transformers import AutoTokenizer
import torch.nn as nn

from model.audio_prompt_generator import AudioPromptGenerator
from model.projector import Projector, AVFusionBlock
from model.evf_sam2 import EvfSam2Model
from model.evf_sam import EvfSamModel
from einops import rearrange

class AVTSAM(nn.Module):
    def __init__(self, args):
        super(AVTSAM, self).__init__()
        self.args = args
        self.depth = 32

        self.tokenizer, self.model = self.init_model()
        self.common_dim = 512
        self.visual_dim = 256
        self.projector = Projector(args.projector_type, audio_embedding_dim=1024, vision_embedding_dim=512, common_dim=self.common_dim)

        self.prompt_feature_up_layer = nn.Linear(self.visual_dim, self.common_dim)
        self.prompt_feature_down_layer = nn.Linear(self.common_dim, self.visual_dim)

        
        if args.use_adapter:
            print(f"Using adapter type: {args.adapter_type}")
            if args.adapter_type == "mul":
                self.audio_prompt_generator = AudioPromptGenerator(depth=self.depth, aud_in_dim=self.common_dim, aud_hidden_dim=self.common_dim)
            else:
                self.audio_prompt_generator = AudioPromptGenerator(depth=self.depth, aud_in_dim=1024, aud_hidden_dim=1024)
        
        self.av_layers = nn.ModuleList()
        for _ in range(4):
            self.av_layers.append(
                AVFusionBlock()
            )

    def init_model(self):
        if self.args.evf_version == "evf_sam":
            print("Using evf_sam")
            version = "YxZhang/evf-sam"
            version = "YxZhang/evf-sam-multitask"
            tokenizer = AutoTokenizer.from_pretrained(
                version,
                padding_side="right",
                use_fast=False,
            )
            torch_dtype = torch.float32
            # torch_dtype = torch.half

            kwargs = {"torch_dtype": torch_dtype}
            model = EvfSamModel.from_pretrained(
                version, low_cpu_mem_usage=True,
                **kwargs
            )
        elif self.args.evf_version == "evf_sam_multitask":
            print("Using evf_sam_multitask")
            version = "YxZhang/evf-sam-multitask"
            tokenizer = AutoTokenizer.from_pretrained(
                version,
                padding_side="right",
                use_fast=False,
            )
            torch_dtype = torch.float32
            # torch_dtype = torch.half

            kwargs = {"torch_dtype": torch_dtype}
            model = EvfSamModel.from_pretrained(
                version, low_cpu_mem_usage=True,
                **kwargs
            )

        elif self.args.evf_version == "evf_sam2":
            self.depth = 48
            version = "YxZhang/evf-sam2"
            version = "YxZhang/evf-sam2-multitask"

            tokenizer = AutoTokenizer.from_pretrained(
                version,
                padding_side="right",
                use_fast=False,
            )
            torch_dtype = torch.float32
            # torch_dtype = torch.half

            kwargs = {"torch_dtype": torch_dtype}
            model = EvfSam2Model.from_pretrained(
                version, low_cpu_mem_usage=False,
                args = self.args,
                **kwargs
            )

        model = model.cuda()
        # model.eval()

        return tokenizer, model

    def set_decoder_trainable(self, turn_on: bool):
        if turn_on:
            print("Training decoder")
            try:
                # evf_sam2
                self.model.visual_model.sam_mask_decoder.train()
            except:
                # evf_sam
                self.model.visual_model.mask_decoder.train()
        else:
            print("Freezing decoder")
        
        

        try:
            for param in self.model.visual_model.sam_mask_decoder.parameters():
                param.requires_grad = turn_on
        except:
            for param in self.model.visual_model.mask_decoder.parameters():
                param.requires_grad = turn_on
        try:
            self.model.visual_model.sam_prompt_encoder.no_mask_embed.requires_grad_(turn_on)
        except:
            self.model.visual_model.prompt_encoder.no_mask_embed.requires_grad_(turn_on)
    
    def fuse_audio_visual_features(self, prompt_feature, visual_feature, visual_pe):
        # Fuse audio and visual features

        # visual_feature = (B * T) C(256) W H 
        # prompt_feature = B T C(512)
        # print(visual_feature.shape)
        # print(visual_pe.shape)
        B, T, C = prompt_feature.shape
        b, c, h, w = visual_feature.shape

    

        visual_feature = visual_feature.reshape(B, T, c, h, w)
        visual_pe = visual_pe.reshape(B, T, c, h, w)

        # prompt_feature = rearrange(prompt_feature, 'b t c -> (b t) c' )
        prompt_feature = self.prompt_feature_down_layer(prompt_feature)
        # b, c, h, w = visual_feature.shape
        # print(f"prompt_feature: {prompt_feature.shape}")
        
        visual_feature = rearrange(visual_feature, 'b t c h w -> b t (h w) c')
        visual_pe = rearrange(visual_pe, 'b t c h w -> b t (h w) c')

        prompt_feature = prompt_feature.unsqueeze(2)

        fused_prompt_feature = prompt_feature
        fused_visual_feature = visual_feature
        
        for layer in self.av_layers:
            fused_visual_feature_list = []
            fused_prompt_feature_list = []
            # fused_prompt_feature = layer(fused_prompt_feature, visual_feature, prompt_feature, visual_pe)
            for b in range(B):
                # print(fused_prompt_feature[b].shape)
                # print(fused_visual_feature[b].shape)
                # print(prompt_feature[b].shape)
                # print(visual_pe[b].shape)
                fused_visual_feature_item, fused_prompt_feature_item = layer(fused_prompt_feature[b], fused_visual_feature[b], prompt_feature[b] , visual_pe[b])
                fused_visual_feature_list.append(fused_visual_feature_item)
                fused_prompt_feature_list.append(fused_prompt_feature_item)
            fused_visual_feature = torch.stack(fused_visual_feature_list)
            fused_prompt_feature = torch.stack(fused_prompt_feature_list)
        

        fused_prompt_feature = fused_prompt_feature + prompt_feature

        fused_prompt_feature = fused_prompt_feature.squeeze(2)
        
        fused_prompt_feature = self.prompt_feature_up_layer(fused_prompt_feature)

        fused_visual_feature = rearrange(fused_visual_feature, 'b t (h w) c -> (b t) c h w', b=B, t=T, h=h, w=w, c=c)
        fused_prompt_feature = rearrange(fused_prompt_feature, 'b t c -> (b t) c', b=B, t=T)


    
        return fused_prompt_feature, fused_visual_feature
        

    def forward(self, images_sam, images_beit, clip_embeddings, clap_embeddings, original_size_lists):

        # compute text embeddings
        encoder_features = self.projector.get_features(clap_embeddings, clip_embeddings)

        B, T, C, H, W = images_sam.shape
        Bb, Tb, Cb, Hb, Wb = images_beit.shape
        images_sam = images_sam.reshape(B*T, C, H, W)
        images_beit = images_beit.reshape(Bb*Tb, Cb, Hb, Wb)
        # get prompt features
        if self.args.use_adapter:
            if self.args.adapter_type == "clap":
                clap_embeddings = clap_embeddings.reshape(B*T, -1)
                clap_embeddings = torch.functional.F.normalize(clap_embeddings, p=2, dim=1)
                prompt_features = self.audio_prompt_generator(clap_embeddings)
            elif self.args.adapter_type == "mul":
                # mul_features = self.projector.get_clip_clap_mul_features()
                mul_features = encoder_features
                mul_features = mul_features.reshape(B*T, self.common_dim)
                prompt_features = self.audio_prompt_generator(mul_features)

        else:
            prompt_features = None
            
        # get image features
        sam_backbone_output = self.model.get_sam_encoder_features(images_sam, prompt_features)


        dense_prompt_embeddings = None
        if self.args.av_fuse:
            # only works for evf_sam2
            sam_image_embeddings = sam_backbone_output["backbone_fpn"][-1]
            sam_pos_enc = sam_backbone_output["vision_pos_enc"][-1]
            encoder_features, dense_prompt_embeddings = self.fuse_audio_visual_features(encoder_features, sam_image_embeddings, sam_pos_enc)

        
        # get text embeddings 
        text_projected_embeddings = self.projector.get_text_embeddings(encoder_features)
        text_projected_embeddings = text_projected_embeddings.reshape(B*T, 3, 1024)        
        
        # [B * T, 1, 1024]        
        beit3_features = self.model.get_beit3_features(images_beit, text_projected_embeddings)
 
        pred_masks = self.get_pred_masks(images_sam, sam_backbone_output,beit3_features, prompt_features, original_size_lists, dense_prompt_embeddings)
        return pred_masks


    def get_pred_masks(self, images, sam_backbone_output,beit3_features ,   prompt_features, original_size_list, dense_prompt_embeddings):


    
        pred_mask = self.model.inference_text_embeddings(
            images,
            sam_backbone_output,
            beit3_features,
            prompt_features,
            resize_list=[None],
            original_size_list=original_size_list,
            dense_prompt_embeddings=dense_prompt_embeddings
        )

   
        return pred_mask 
