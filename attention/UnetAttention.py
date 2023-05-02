from fastmri.models.unet import Unet, ConvBlock, TransposeConvBlock
from fastmri.pl_modules import UnetModule
import torch
from torch import nn
from torch.nn import functional as F
from torchgeometry.losses import SSIM


class UnetAttention(Unet):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        attention_only: bool = False,
    ):
        super().__init__(in_chans = in_chans, out_chans = out_chans, chans = chans, num_pool_layers = num_pool_layers, drop_prob = drop_prob)

        #print("UnetAttention parameters: ")
        #print("in_chans", in_chans)
        #print("out_chans", out_chans)
        #print("chans", chans)
        #print("num_pool_layers", num_pool_layers)
        #print("drop_prob", drop_prob)

        if attention_only:
            # freeze all layers except attention layers
            for param in self.parameters():
                param.requires_grad = False

        ch = chans * (2 ** (num_pool_layers - 1))
        self.up_attention = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_attention.append(AttentionBlock(f_g = ch, f_l = ch, f_int = ch // 2))
            ch //= 2
        self.up_attention.append(AttentionBlock(f_g = ch, f_l = ch, f_int = ch // 2))




    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, attention, conv  in zip(self.up_transpose_conv, self.up_attention, self.up_conv):            
            downsample_layer = stack.pop()
            transp_output = transpose_conv(output)
            attention_output = attention(g=transp_output, x=downsample_layer)

        #    # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([attention_output, transp_output], dim=1)
            output = conv(output)

        return output
    
class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        
        self.w_g = nn.Sequential(
                                nn.Conv2d(f_g, f_int,
                                         kernel_size=1, stride=1,
                                         padding=0, bias=True),
                                nn.BatchNorm2d(f_int)
        )
        
        self.w_x = nn.Sequential(
                                nn.Conv2d(f_l, f_int,
                                         kernel_size=1, stride=1,
                                         padding=0, bias=True),
                                nn.BatchNorm2d(f_int)
        )
        
        self.psi = nn.Sequential(
                                nn.Conv2d(f_int, 1,
                                         kernel_size=1, stride=1,
                                         padding=0,  bias=True),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        
        return psi*x        