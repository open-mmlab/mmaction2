import torch
import torch.nn as nn
from clip.model import Transformer
from mmengine.model import BaseModule


class TransformerAdapter(BaseModule):

    def __init__(self,
                 clip_model: nn.Module,
                 num_segs: int,
                 num_layers: int = 6):
        super(TransformerAdapter, self).__init__()
        self.num_segs = num_segs

        embed_dim = clip_model.text_projection.shape[1]
        transformer_width = clip_model.ln_final.weight.shape[0]
        transformer_heads = transformer_width // 64

        self.frame_position_embeddings = nn.Embedding(self.num_segs, embed_dim)
        self.transformer = Transformer(
            width=embed_dim, layers=num_layers, heads=transformer_heads)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        b, seq_length, c = x.size()

        x_original = x
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=x.device)
        embeddings = self.frame_position_embeddings(position_ids)
        x = x + embeddings.unsqueeze(0)
        x = x.transpose(0, 1)  # NLD -> LND
        x = self.transformer(x)
        x = x.transpose(0, 1)  # LND -> NLD
        x = x.type(x_original.dtype) + x_original
        return x.mean(dim=1)
