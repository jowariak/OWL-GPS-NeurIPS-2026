"""Original dual MMSeg FCN segmentation heads (optional MMSeg dependency)."""
import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels
        self.norm_cfg=dict(type="BN", requires_grad=True)
        conv_padding = (kernel_size // 2) * dilation
        convs = []
        for i in range(num_convs):
            _in_channels = self.in_channels if i == 0 else self.channels
            convs.append(
                ConvModule(
                    _in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        if len(convs) == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output

class head(nn.Module):
    def __init__(self, output_embed_dim = 768, num_classes = 1):
        """
        Takes fused features and processes them through two FCN heads.

        Arguments:
        - output_embed_dim: Number of feature channels in the fused feature map.
        - num_classes: Number of output classes for segmentation.
        """
        super(head, self).__init__()

        # First segmentation head
        self.head = FCNHead(
            in_channels=output_embed_dim,
            channels=256,
            num_classes=num_classes,
            num_convs=1,
            in_index=-1,
            concat_input=False,
            dropout_ratio=0.1,
            align_corners=False
        )

        # Second segmentation head
        self.head2 = FCNHead(
            in_channels=output_embed_dim,
            channels=256,
            num_classes=num_classes,
            num_convs=2,
            in_index=-1,
            concat_input=False,
            dropout_ratio=0.1,
            align_corners=False
        )

    def forward(self, fused_features):
        """
        Processes fused features through both segmentation heads.

        Arguments:
        - fused_features: Tensor of shape (B, output_embed_dim, 256, 256).

        Returns:
        - output1: Output from first segmentation head.
        - output2: Output from second segmentation head.
        """
        output1 = self.head(fused_features)  # Segmentation output from head 1
        output2 = self.head2(fused_features)  # Segmentation output from head 2

        return output1, output2

DualSegmentationHead = head
