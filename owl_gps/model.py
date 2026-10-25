"""Concept-specific frozen encoders, relevance weighting and dual-head prediction."""
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from .relevance import orthogonalize_concepts, sample_relevance


class OWLGPSModel(nn.Module):
    """Pluggable concept-backbone OWL-GPS model.

    encoders: one module per input channel, each consuming (B,1,1,H,W)
      for the original TemporalViTEncoder (or adapt it for another backbone).
    neck: turns each encoder output into (B,feature_dim,H,W).
    relevance_encoder: takes (B,K,feature_dim,H,W), returns (mu,sigma).
    decoder: takes a 1-tuple containing the fused map; returns two logit maps.
    """

    def __init__(self, encoders: nn.ModuleList, neck: nn.Module,
                 relevance_encoder: nn.Module, decoder: nn.Module,
                 *, freeze_encoders: bool = True, orthogonalize: bool = True):
        super().__init__()
        self.encoders = encoders
        self.neck = neck
        self.relevance_encoder = relevance_encoder
        self.decoder = decoder
        self.orthogonalize = orthogonalize
        if freeze_encoders:
            for module in self.encoders:
                module.requires_grad_(False)
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        for encoder in self.encoders:
            if not any(p.requires_grad for p in encoder.parameters()):
                encoder.eval()
        return self

    def forward(self, images: torch.Tensor, *, stochastic: bool | None = None):
        if images.ndim != 5 or images.shape[1] != len(self.encoders):
            raise ValueError("Expected images as (B,num_concepts,1,H,W)")
        features = []
        for i, encoder in enumerate(self.encoders):
            # The original ViT uses Conv3D and expects an explicit T=1 dimension.
            x = images[:, i].unsqueeze(2)
            if any(p.requires_grad for p in encoder.parameters()):
                tokens = encoder(x)
            else:
                with torch.no_grad():
                    tokens = encoder(x)
            map_out = self.neck(tokens)
            if isinstance(map_out, (tuple, list)):
                map_out = map_out[0]
            features.append(map_out)
        embeddings = torch.stack(features, dim=1)
        if self.orthogonalize:
            embeddings = orthogonalize_concepts(embeddings)
        mu, sigma = self.relevance_encoder(embeddings)
        if stochastic is None:
            stochastic = self.training
        rel = sample_relevance(mu, sigma, stochastic=stochastic)
        fused = (embeddings * rel[:, :, None, None, None]).sum(dim=1)
        logits_a, logits_b = self.decoder((fused,))
        return logits_a, logits_b, mu, sigma

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def load_component_checkpoint(model: OWLGPSModel, checkpoint: str | Path):
    """Load weights saved by the original research script, if compatible."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # Original checkpoint contains these three keys; frozen encoders load separately.
    model.relevance_encoder.load_state_dict(state["relevance_encoder"])
    model.decoder.load_state_dict(state["segmentation_decoder"])
    model.neck.load_state_dict(state["neck"])
    return model


def build_original_2019_model(checkpoint_paths, *, checkpoint=None):
    """Build the 33-channel architecture; pretrained weights are supplied by caller.

    Optional dependencies: timm, mmcv and mmseg versions compatible with the
    source checkpoint. This construction is computationally demanding.
    """
    from .backbone import TemporalViTEncoder, ConvTransformerTokensToEmbeddingNeck
    from .heads import DualSegmentationHead
    from .relevance import RelevanceEncoder2

    paths = [str(p) for p in checkpoint_paths]
    if len(paths) != 33:
        raise ValueError(f"2019 model requires 33 concept checkpoint files; got {len(paths)}")
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing concept checkpoint: {path}")
    encoders = nn.ModuleList([
        TemporalViTEncoder(img_size=256, patch_size=16, num_frames=1,
                           embed_dim=768, pretrained=p) for p in paths
    ])
    model = OWLGPSModel(
        encoders=encoders,
        neck=ConvTransformerTokensToEmbeddingNeck(
            embed_dim=768, output_embed_dim=768,
            drop_cls_token=True, Hp=16, Wp=16
        ),
        relevance_encoder=RelevanceEncoder2(embed_dim=768),
        decoder=DualSegmentationHead(output_embed_dim=768),
    )
    if checkpoint is not None:
        load_component_checkpoint(model, checkpoint)
    return model
