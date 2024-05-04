from dataclasses import dataclass
from typing import List, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential


# Macros.

def Convolution1x1(input_channels: int, output_channels: int) -> nn.Module:

    return nn.Conv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=1,
        stride=1,
        padding=0,
    )


def Convolution3x3(input_channels: int, output_channels: int) -> nn.Module:

    return nn.Conv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )


def Convolution4x4(input_channels: int, output_channels: int) -> nn.Module:

    return nn.Conv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=False,
    )


def Normalization(channels: int) -> nn.Module:

    return nn.GroupNorm(
        num_groups=min(channels, 32),
        num_channels=channels,
    )


def Repeat(module, channels_list: List[int]) -> nn.Module:

    return nn.Sequential(*(
        module(
            input_channels=input_channels,
            output_channels=output_channels,
        ) for input_channels, output_channels in zip(
            channels_list[: -1],
            channels_list[1 :],
        )
    ))


# Modules.

class ResidualBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.normalization = Normalization(channels=channels)

        self.convolution = Convolution3x3(
            input_channels=channels,
            output_channels=channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        z = self.normalization(x)
        z = F.leaky_relu(z)
        z = self.convolution(z)

        return x + z


class ResNetBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.residual_block_1 = ResidualBlock(channels=channels)
        self.residual_block_2 = ResidualBlock(channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.residual_block_1(x)
        x = self.residual_block_2(x)

        return x


class UpsampleBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.normalization = Normalization(channels=input_channels)

        self.convolution = Convolution3x3(
            input_channels=input_channels,
            output_channels=output_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.normalization(x)
        x = F.leaky_relu(x)
        x = self.upsample(x)
        x = self.convolution(x)

        return x


class DownsampleBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.normalization = Normalization(channels=input_channels)

        self.convolution = Convolution4x4(
            input_channels=input_channels,
            output_channels=output_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.normalization(x)
        x = F.leaky_relu(x)
        x = self.convolution(x)

        return x


class AttentionBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.normalization = Normalization(channels=channels)

        self.convolution_1 = Convolution1x1(
            input_channels=channels,
            output_channels=channels,
        )

        self.convolution_2 = Convolution1x1(
            input_channels=channels,
            output_channels=channels,
        )

        self.convolution_3 = Convolution1x1(
            input_channels=channels,
            output_channels=channels,
        )

        self.convolution_4 = Convolution1x1(
            input_channels=channels,
            output_channels=channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape

        z = self.normalization(x)
        q = self.convolution_1(z)
        k = self.convolution_2(z)
        v = self.convolution_3(z)

        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')  # Transposed.
        v = rearrange(v, 'b c h w -> b (h w) c')

        z = F.softmax(q @ k, dim=-1) @ v
        z = rearrange(z, 'b (h w) c -> b c h w', h=H, w=W)
        z = self.convolution_4(z)

        return x + z


class UpBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.resnet_block = ResNetBlock(channels=input_channels)

        self.upsample_block = UpsampleBlock(
            input_channels=input_channels,
            output_channels=output_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.resnet_block(x)
        x = self.upsample_block(x)

        return x


class DownBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) ->  None:
        super().__init__()

        self.resnet_block = ResNetBlock(channels=input_channels)

        self.downsample_block = DownsampleBlock(
            input_channels=input_channels,
            output_channels=output_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.resnet_block(x)
        x = self.downsample_block(x)

        return x


class MiddleBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.resnet_block_1 = ResNetBlock(channels=channels)
        self.resnet_block_2 = ResNetBlock(channels=channels)
        self.attention_block = AttentionBlock(channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.resnet_block_1(x)
        x = self.attention_block(x)
        x = self.resnet_block_2(x)

        return x


class Encoder(nn.Module):

    def __init__(self, channels_list: List[int]) -> None:
        super().__init__()

        self.down_blocks = Repeat(module=DownBlock, channels_list=channels_list)
        self.middle_block = MiddleBlock(channels=channels_list[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.down_blocks(x)
        x = self.middle_block(x)

        return x


class Decoder(nn.Module):

    def __init__(self, channels_list: List[int]) -> None:
        super().__init__()

        self.up_blocks = Repeat(module=UpBlock, channels_list=channels_list)
        self.middle_block = MiddleBlock(channels=channels_list[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.middle_block(x)
        x = self.up_blocks(x)

        return x


class GaussianDistribution(nn.Module):

    def __init__(self, parameters: torch.Tensor) -> None:
        super().__init__()

        self.mean, self.log_variance = parameters.chunk(chunks=2, dim=1)

    def sample(self) -> torch.Tensor:

        epsilon = torch.randn_like(self.mean, device=self.mean.device)
        standard_deviation = torch.exp(0.5 * self.log_variance)
        x = epsilon * standard_deviation + self.mean

        return x


# Models.

@dataclass(frozen=True)
class VAEOptions:
    input_channels: int
    output_channels: int
    latent_channels: int
    encoder_channels_list: List[int]
    decoder_channels_list: List[int]


class VAE(nn.Module):

    def __init__(self, options: VAEOptions) -> None:
        super().__init__()

        self.encoder = Encoder(channels_list=options.encoder_channels_list)
        self.decoder = Decoder(channels_list=options.decoder_channels_list)

        # Input to encoder.

        self.convolution_1 = Convolution3x3(
            input_channels=options.input_channels,
            output_channels=options.encoder_channels_list[0],
        )

        # Encoder to latent.

        self.convolution_2 = Convolution3x3(
            input_channels=options.encoder_channels_list[-1],
            output_channels=options.latent_channels * 2,
        )

        # Latent to decoder.

        self.convolution_3 = Convolution3x3(
            input_channels=options.latent_channels,
            output_channels=options.decoder_channels_list[0],
        )

        # Decoder to output.

        self.convolution_4 = Convolution3x3(
            input_channels=options.decoder_channels_list[-1],
            output_channels=options.output_channels,
        )

    def encode(self, x: torch.Tensor) -> GaussianDistribution:

        x = self.convolution_1(x)
        x = self.encoder(x)
        x = self.convolution_2(x)

        distribution = GaussianDistribution(x)

        return distribution

    def decode(self, z: torch.Tensor) -> torch.Tensor:

        x = self.convolution_3(z)
        x = self.decoder(x)
        x = self.convolution_4(x)
        x = torch.sigmoid(x)

        return x

    def forward(self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, GaussianDistribution]:

        distribution = self.encode(x)
        z = distribution.sample()
        x = self.decode(z)

        return x, z, distribution
