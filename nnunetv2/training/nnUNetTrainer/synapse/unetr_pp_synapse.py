from torch import nn
import torch
import numpy as np
from typing import Tuple, Union, List
from nnunetv2.training.nnUNetTrainer.synapse.clip_utils import FeatureNet
from nnunetv2.training.nnUNetTrainer.synapse.neural_network import SegmentationNetwork
from nnunetv2.training.nnUNetTrainer.synapse.dynunet_block import UnetOutBlock, UnetResBlock
from nnunetv2.training.nnUNetTrainer.synapse.model_components import UnetrPPEncoder, UnetrUpBlock


class UNETR_PP(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: List,
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv2d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.

        Examples::

            # for single channel input 4-channel output with patch size of (64, 128, 128), feature size of 16, batch
            norm and depths of [3, 3, 3, 3] with output channels [32, 64, 128, 256], 4 heads, and 14 classes with
            deep supervision:
            >>> net = UNETR_PP(in_channels=1, out_channels=14, img_size=(64, 128, 128), feature_size=16, num_heads=4,
            >>>                 norm_name='batch', depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True)
        """

        super().__init__()
        self.mul_factor = int((img_size[0]*img_size[1])/(128*128))
        if depths is None:
            depths = [3, 3, 3, 3]
        dims = [feature_size*2, feature_size*4, feature_size*8, feature_size*16]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(
                f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (4, 4)
        self.feat_size = (
            # 8 is the downsampling happened through the four encoders stages
            img_size[0] // self.patch_size[0] // 8,
            # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,
        )
        self.hidden_size = hidden_size

        self.unetr_pp_encoder = UnetrPPEncoder(
            dims=dims, depths=depths, num_heads=num_heads, mul_factor=self.mul_factor)

        self.encoder1 = UnetResBlock(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8 * 8 * self.mul_factor,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 16 * self.mul_factor,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32 * 32 * self.mul_factor,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4),
            norm_name=norm_name,
            out_size=512*512,
            conv_decoder=True,
        )
        self.radio_feature_net = FeatureNet()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.out1 = UnetOutBlock(
            spatial_dims=2, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(
                spatial_dims=2, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(
                spatial_dims=2, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x_in, feature=None):
        if feature is not None:
            radio_feature = self.radio_feature_net(feature)
        img_feature, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        # print('x_output shape', x_output.shape)
        # print('enc1 enc2 enc3 enc4', enc1.shape,
        #       enc2.shape, enc3.shape, enc4.shape)

        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)
        if feature is not None:
            # print('radio_feature, img_feature, self.logit_scale.exp() is', radio_feature, img_feature, self.logit_scale.exp())
            return logits, radio_feature, img_feature, self.logit_scale.exp()
        else:
            return logits
