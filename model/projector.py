import torch
import torch.nn as nn
import math
from typing import Any, Dict, List, Tuple, Type, Optional
import torch.nn.functional as F



class Projector(nn.Module):
    def __init__(self, type = "clap", audio_embedding_dim=1024, vision_embedding_dim=512, common_dim=512):
        super(Projector, self).__init__()

        self.common_dim = common_dim
        self.text_embedding_shape = (3, 1024)
        self.project_type = type

        self.clip_clap_mul_features = None
        self.features = None

        
        # Projection layers to map embeddings to a common dimension
        self.audio_proj = nn.Linear(audio_embedding_dim, common_dim)
        self.vision_proj = nn.Linear(vision_embedding_dim, common_dim)
        # # Self-attention layers
        # self.self_attention_audio = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads)
        # self.self_attention_vision = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads)
        
        # # Cross-attention layers
        # self.cross_attention_audio_to_vision = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads)
        # self.cross_attention_vision_to_audio = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads)
        
        # Projection layer to map combined embeddings to the LLM text embedding space
        text_embedding_dim = self.text_embedding_shape[1]  # 1024 in this case
        text_embedding_channels = self.text_embedding_shape[0]  # 3 in this case
        target_dim = text_embedding_dim * text_embedding_channels
        print(f"Using {type} projector")
        # if type == "default":
        #     print("default projector")
        #     # fusion layers
        #     self.layers = nn.ModuleList()
        #     for _ in range(4):
        #         self.layers.append(
        #             AVFusionBlock()
        #     )
        #     self.to_text_space = nn.Sequential(
        #         nn.Linear(common_dim * 2, common_dim * 2),
        #         nn.ReLU(),
        #         nn.Linear(common_dim * 2, target_dim),
                
        #     )
        #     # self.to_text_space = nn.Linear(common_dim * 2, text_embedding_dim * text_embedding_channels)

        # if type == "clap":
        #     self.to_text_space = nn.Sequential(
        #             nn.Linear(common_dim, common_dim),
        #             nn.ReLU(),
        #             nn.Linear(common_dim, target_dim),
        #     )
        # elif type == "clip":
        #     self.to_text_space = nn.Sequential(
        #             nn.Linear(common_dim, common_dim),
        #             nn.ReLU(),
        #             nn.Linear(common_dim, target_dim),
        #     )
        # elif type == "concat":
        #     self.to_text_space = nn.Sequential(
        #             nn.Linear(common_dim * 2, common_dim * 2),
        #             nn.ReLU(),
        #             nn.Linear(common_dim * 2, target_dim),
        #     )
        # else:
        self.to_text_space = nn.Sequential(
                nn.Linear(common_dim,  common_dim),
                nn.ReLU(),
                nn.Linear(common_dim , target_dim),
        )
            
            
            
    def get_features(self, audio_embeddings :torch.Tensor, vision_embeddings: torch.Tensor) -> torch.Tensor:

        if self.project_type == "clap":
            features = self.audio_proj(audio_embeddings)
        elif self.project_type == "clip":
            features = self.vision_proj(vision_embeddings)
        elif self.project_type == "mul":
            # normalize the embeddings
            audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
            vision_embeddings = F.normalize(vision_embeddings, p=2, dim=-1)

            projected_audio = self.audio_proj(audio_embeddings)
            # projected_vision = self.vision_proj(vision_embeddings)

            features = projected_audio * vision_embeddings
            # get the elementwise product of the embeddings
            # self.clip_clap_mul_features = mul_features
            
        return features
    
    def get_text_embeddings(self, features):
        return self.to_text_space(features)
     


    def forward(self, audio_embeddings :torch.Tensor, vision_embeddings: torch.Tensor) -> torch.Tensor:

        if self.project_type == "clap":
            # if self.norm:
            #     audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
            features = self.audio_proj(audio_embeddings)
        elif self.project_type == "clip":
            features = self.vision_proj(vision_embeddings)
        elif self.project_type == "mul":
            # normalize the embeddings
            # if self.norm:
            audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
            vision_embeddings = F.normalize(vision_embeddings, p=2, dim=-1)

            projected_audio = self.audio_proj(audio_embeddings)
            projected_vision = self.vision_proj(vision_embeddings)
            features = projected_audio * projected_vision
            # get the elementwise product of the embeddings
            # self.clip_clap_mul_features = mul_features
            
        return features
    
    def get_clip_clap_mul_features(self):
        return self.clip_clap_mul_features

 
class AVFusionBlock(nn.Module):
    def __init__(self, num_heads=8,   common_dim=256):
        super(AVFusionBlock, self).__init__()

        # self.prompt_embed_dim = prompt_embed_dim
        self.common_dim = common_dim
        self.num_heads = num_heads

        self.embed_vis = MLPBlock(self.common_dim, self.common_dim)
        self.embed_audio = MLPBlock(self.common_dim, self.common_dim)
        self.embed_audio2 = MLPBlock(self.common_dim, self.common_dim)
        
        self.embed_av = MLPBlock(self.common_dim, self.common_dim)

        self.avt_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=self.num_heads, dropout=0.1)
        self.avs_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=self.num_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(self.common_dim) 
        self.norm2 = nn.LayerNorm(self.common_dim) 
        self.norm3_1 = nn.LayerNorm(self.common_dim) 
        self.norm3_2 = nn.LayerNorm(self.common_dim) 
        self.norm4 = nn.LayerNorm(self.common_dim) 
        self.norm5_1 = nn.LayerNorm(self.common_dim) 
        self.norm5_2 = nn.LayerNorm(self.common_dim) 
        self.norm6 = nn.LayerNorm(self.common_dim) 
        
    
    def forward(self, audio_feature, visual_feature, audio_pe, visual_pe):
        
        b, n_hw, c = visual_feature.shape

        audio_feature = audio_feature + self.embed_audio(audio_feature)
        audio_feature = self.norm1(audio_feature)

        visual_feature = visual_feature + self.embed_vis(visual_feature)
        visual_feature = self.norm2(visual_feature)

        # Temporal attn
        audio_feature_pe = audio_feature + audio_pe
        visual_feature_pe = visual_feature + visual_pe

        avt_audio_attn, avt_audio_attn_weight = self.avt_attention(visual_feature_pe, audio_feature_pe.repeat(1, n_hw, 1), audio_feature.repeat(1, n_hw, 1)) # B HW C
        avt_audio_attn = torch.nn.AdaptiveAvgPool1d(1)(avt_audio_attn.transpose(1, 2)).transpose(1, 2)
        fused_audio_feature = audio_feature + avt_audio_attn # B, 1, C
        fused_audio_feature = self.norm3_1(fused_audio_feature)

        # Spatial attn
        fused_audio_feature_pe = fused_audio_feature + audio_pe
        visual_feature_pe = visual_feature + visual_pe
        avs_audio_attn, avs_audio_attn_weight = self.avs_attention(visual_feature_pe, fused_audio_feature_pe, fused_audio_feature) # B HW C
        avs_audio_attn = torch.nn.AdaptiveAvgPool1d(1)(avs_audio_attn.transpose(1, 2)).transpose(1, 2)
        fused_audio_feature = fused_audio_feature + avs_audio_attn # B 1 C
        fused_audio_feature = self.norm3_2(fused_audio_feature)
        
        # MLP block
        fused_audio_feature = fused_audio_feature + self.embed_audio2(fused_audio_feature)
        fused_audio_feature = self.norm4(fused_audio_feature)
        
        # Attention with PE features

        # Temporal attn
        visual_feature_pe = visual_feature + visual_pe
        fused_audio_feature_pe = fused_audio_feature + audio_pe
        avt_visual_attn, avt_visual_attn_weight = self.avt_attention(fused_audio_feature_pe.repeat(1, n_hw, 1), visual_feature_pe, visual_feature) # B HW C
        fused_visual_feature = visual_feature + avt_visual_attn
        fused_visual_feature = self.norm5_1(fused_visual_feature)

        # Spatial
        fused_visual_feature_pe = fused_visual_feature + visual_pe
        fused_audio_feature_pe = fused_audio_feature + audio_pe
        avs_visual_attn, avs_visual_attn_weight = self.avs_attention(fused_audio_feature_pe, fused_visual_feature_pe, fused_visual_feature) # B 1 C
        fused_visual_feature = fused_visual_feature + avs_visual_attn.repeat(1, n_hw, 1)
        fused_visual_feature = self.norm5_2(fused_visual_feature)

        # MLP block
        fused_visual_feature = fused_visual_feature + self.embed_av(fused_visual_feature)
        fused_visual_feature = self.norm6(fused_visual_feature)

        return fused_visual_feature, fused_audio_feature
        # return fused_audio_feature
    

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

        
