import math
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, type="clap", audio_embedding_dim=1024, vision_embedding_dim=512, common_dim=512):
        super(Projector, self).__init__()

        self.common_dim = common_dim
        self.text_embedding_shape = (3, 1024)
        self.project_type = type

        self.clip_clap_mul_features = None
        self.features = None

        # Projection layers to map embeddings to a common dimension
        self.audio_proj = nn.Linear(audio_embedding_dim, common_dim)
        self.vision_proj = nn.Linear(vision_embedding_dim, common_dim)

        # Projection layer to map combined embeddings to the LLM text embedding space
        text_embedding_dim = self.text_embedding_shape[1]  # 1024 in this case
        text_embedding_channels = self.text_embedding_shape[0]  # 3 in this case
        target_dim = text_embedding_dim * text_embedding_channels
        print(f"Using {type} projector")
        self.to_text_space = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.ReLU(),
            nn.Linear(common_dim, target_dim),
        )

        self.__reset_parameters()

    def get_features(self, audio_embeddings: torch.Tensor, vision_embeddings: torch.Tensor) -> torch.Tensor:

        if self.project_type == "clap":
            audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
            features = self.audio_proj(audio_embeddings)
        elif self.project_type == "clip":
            vision_embeddings = F.normalize(vision_embeddings, p=2, dim=-1)
            features = self.vision_proj(vision_embeddings)
        elif self.project_type == "mul":
            # normalize the embeddings
            audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
            vision_embeddings = F.normalize(vision_embeddings, p=2, dim=-1)

            projected_audio = self.audio_proj(audio_embeddings)
            projected_vision = self.vision_proj(vision_embeddings)

            features = projected_audio * projected_vision
        return features

    def get_text_embeddings(self, features):
        return self.to_text_space(features)

    def __reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
