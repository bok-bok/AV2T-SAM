import torch.nn as nn
import math
from torch.nn.init import trunc_normal_


class AudioPromptGenerator(nn.Module):
    def __init__(self,  depth, aud_in_dim=1024, aud_hidden_dim=1024):
        """
        Args:
        """
        super(AudioPromptGenerator, self).__init__()
        self.depth = depth
        self.aud_in_dim = aud_in_dim
        self.aud_hidden_dim = aud_hidden_dim

        

        # self.shared_mlp = nn.Linear(self.aud_hidden_dim, self.embed_dim)
        # self.embedding_generator = nn.Linear(self.embed_dim, self.embed_dim//self.scale_factor)
        channels = [144, 144, 144, 
                    288, 288, 288, 288, 288, 288, 
                    576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 
                    576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 
                    576, 576, 576, 576, 576, 576, 
                    1152, 1152, 1152]
        if self.depth == 48:
            # evf_sam2 
            for i in range(self.depth):
                lightweight_mlp = nn.Sequential(
                    nn.Linear(self.aud_in_dim, self.aud_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.aud_hidden_dim, channels[i]),
                )
                setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)
        else:
            # evf_sam
            for i in range(self.depth):
                lightweight_mlp = nn.Sequential(
                    nn.Linear(self.aud_in_dim, self.aud_hidden_dim),
                    nn.GELU(),
                )
                setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)
            self.shared_mlp = nn.Linear(self.aud_hidden_dim, 1280)


                

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, feature):
        # feature = feature.squeeze(1)
        B, C = feature.shape             # C: 1024
        prompts = []
        if self.depth == 48:
            for i in range(self.depth):
                lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))
                prompt = lightweight_mlp(feature)
                prompts.append(prompt)
        else:
            for i in range(self.depth):
                lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))
                prompt = lightweight_mlp(feature)
                prompts.append(self.shared_mlp(prompt))
            
        return prompts
